// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the install record (v1), the "Apps & features" entry and the restore / uninstall engine.
// Written for this project.
#include "record.h"

#include <array>

#include "detect.h"
#include "gamerules.h"
#include "json_writer.h"
#include "launcher.h"
#include "leftovers.h"
#include "optiscaler.h"
#include "payload.h"
#include "rr.h"
#include "rrconfig.h"

namespace wg {

wstring StateDir(const wstring& dir) { return Join(dir, kStateDirName); }
wstring RecordPath(const wstring& dir) { return Join(StateDir(dir), L"install.record"); }
wstring BackupPath(const wstring& dir, const wstring& rel) { return Join(Join(StateDir(dir), L"backup"), rel); }
wstring LegacyStateDir(const wstring& dir) { return Join(dir, kLegacyStateDir); }
wstring LegacyRecordPath(const wstring& dir) { return Join(LegacyStateDir(dir), L"install.manifest"); }
wstring LegacyBackupPath(const wstring& dir, const wstring& rel) { return Join(Join(LegacyStateDir(dir), L"backup"), rel); }

// rc6 L1: a game folder moved (Steam's "Move install folder", another drive letter, a folder renamed): what changed between
// the folder the record names and the folder it is in now. The trailing folder names both share stay (common\<game>\bin\x64);
// the rest is the changed part: from (in the old path) and to (in the new one). False: nothing to change.
static bool MovedPrefix(const wstring& oldDir, const wstring& newDir, wstring& from, wstring& to) {
    auto trim = [](wstring s) { while (s.size() > 3 && (s.back() == L'\\' || s.back() == L'/')) s.pop_back(); return s; };
    from = trim(FullPath(oldDir)); to = trim(FullPath(newDir));
    if (from.empty() || to.empty() || IEq(from, to)) return false;
    while (from.find(L'\\') != wstring::npos && to.find(L'\\') != wstring::npos && IEq(BaseName(from), BaseName(to)) &&
           !BaseName(from).empty()) {
        from = Parent(from); to = Parent(to);
    }
    return !from.empty() && !to.empty();
}
// p with the changed part of the path put in (p unchanged when it is not in the old folder).
static wstring Rebased(const wstring& p, const wstring& from, const wstring& to) {
    if (p.size() < from.size() || _wcsnicmp(p.c_str(), from.c_str(), from.size()) != 0) return p;
    if (p.size() > from.size() && p[from.size()] != L'\\' && p[from.size()] != L'/') return p;  // G:\Game2 is not in G:\Game
    return to + p.substr(from.size());
}
// The record in RecordPath(dir) names another folder: its absolute paths into that folder (game, exe, the OptiScaler.ini
// values of the inikey journal) moved to dir; movedFrom keeps the old folder. The model file and the game's Engine.ini fix
// (rrconfig) are outside the game folder and stay. In memory: SaveRebasedRecord writes it.
static void RebaseRecord(Record& m, const wstring& dir) {
    wstring from, to;
    if (!MovedPrefix(m.game, dir, from, to)) return;
    m.movedFrom = m.game;
    m.game = dir;
    if (!m.exe.empty()) m.exe = Rebased(m.exe, from, to);
    for (auto& c : m.inikeys) {
        if (!c.absent) c.original = ToUtf8(Rebased(FromUtf8(c.original), from, to));
        c.ours = ToUtf8(Rebased(FromUtf8(c.ours), from, to));
    }
}

bool LoadRecord(const wstring& path, Record& m) {
    string raw;
    if (!ReadAll(path, raw)) return false;
    wstring text = FromUtf8(raw);
    size_t pos = 0; bool header = false;
    std::vector<std::array<wstring, 3>> owners;
    while (pos < text.size()) {
        size_t nl = text.find(L'\n', pos);
        wstring line = text.substr(pos, nl == wstring::npos ? wstring::npos : nl - pos);
        pos = nl == wstring::npos ? text.size() : nl + 1;
        while (!line.empty() && line.back() == L'\r') line.pop_back();
        if (line.empty()) continue;
        if (!header) {
            if (line.rfind(L"WRONGGPU-RECORD", 0) == 0) { m.legacy = false; m.format = _wtoi(line.c_str() + 15) >= 2 ? 2 : 1; }
            else if (line.rfind(L"DLSS45AMD-MANIFEST", 0) == 0) m.legacy = true;
            else return false;
            header = true;
            continue;
        }
        std::vector<wstring> f; size_t s = 0;
        for (int i = 0; i < 5; ++i) {
            size_t t = line.find(L'\t', s);
            if (t == wstring::npos || i == 4) { f.push_back(line.substr(s)); break; }
            f.push_back(line.substr(s, t - s)); s = t + 1;
        }
        if (!m.legacy && line.rfind(L"inikey\t", 0) == 0) {  // six fields: split the whole line
            std::vector<wstring> p;
            for (size_t a = 0;;) { const size_t b = line.find(L'\t', a); p.push_back(line.substr(a, b == wstring::npos ? wstring::npos : b - a)); if (b == wstring::npos) break; a = b + 1; }
            if (p.size() == 6 && (p[4] == L"absent" || p[4].rfind(L"=", 0) == 0) && p[5].rfind(L"=", 0) == 0) {
                IniKeyChange c; c.file = p[1]; c.section = ToUtf8(p[2]); c.key = ToUtf8(p[3]);
                c.absent = p[4] == L"absent"; if (!c.absent) c.original = ToUtf8(p[4].substr(1));
                c.ours = ToUtf8(p[5].substr(1));
                m.inikeys.push_back(c);
            } else m.extra.push_back(line);
            continue;
        }
        if (f.size() == 2) {
            if (f[0] == L"version") m.version = f[1];
            else if (f[0] == L"game") m.game = f[1];
            else if (f[0] == L"exe") m.exe = f[1];
            else if (f[0] == L"status") m.status = f[1];
            else if (f[0] == L"regkey") m.regkey = f[1];
            else if (f[0] == L"model") m.model = f[1];
            else if (f[0] == L"model-m") m.modelM = f[1];
            else if (!m.legacy) m.extra.push_back(line);
        } else if (f.size() == 5 && (f[0] == L"copy" || f[0] == L"disable" || (!m.legacy && (f[0] == L"rename" || f[0] == L"dir")))) {
            Entry e; e.action = f[0]; e.existed = f[1] == L"1"; e.orig = f[2] == L"-" ? L"" : f[2]; e.inst = f[3] == L"-" ? L"" : f[3]; e.rel = f[4];
            m.e.push_back(e);
        } else if (!m.legacy && f[0] == L"payload" && f.size() == 5) {
            auto dash = [](const wstring& s) { return s == L"-" ? wstring() : s; };
            // the fifth field holds the rest of the line: the menu plugin's hash, then (rc6) the RR plugin's
            const size_t t = f[4].find(L'	');
            m.payloadVersion = f[1]; m.runtimeSha = dash(f[2]); m.frontEndSha = dash(f[3]); m.pluginSha = dash(f[4].substr(0, t));
            if (t != wstring::npos) { const wstring rest = f[4].substr(t + 1); m.rrPluginSha = dash(rest.substr(0, rest.find(L'	'))); }
        } else if (!m.legacy && f[0] == L"proxy" && f.size() >= 2) {
            m.proxy = f[1]; m.proxyReason = f.size() >= 3 ? f[2] : L"";
        } else if (!m.legacy && f[0] == L"coexist" && f.size() == 4) {
            m.coexist.push_back({f[1], f[2], f[3]});
        } else if (!m.legacy && f[0] == L"rrconfig" && f.size() == 5 && (f[3] == L"absent" || f[3].rfind(L"=", 0) == 0)) {
            RrConfigFix& x = m.rrconfig;
            x.present = true; x.path = f[1];
            auto has = [&](wchar_t c) { return f[2].find(c) != wstring::npos; };
            x.created = has(L'c'); x.sectionAdded = has(L's'); x.wasReadOnly = has(L'w'); x.setReadOnly = has(L'r'); x.backup = has(L'b');
            x.prevAbsent = f[3] == L"absent"; x.prev = x.prevAbsent ? string() : ToUtf8(f[3].substr(1));
            x.folder = f[4] == L"-" ? wstring() : f[4];
        } else if (!m.legacy && f[0] == L"owner" && f.size() == 4) {
            owners.push_back({f[1], f[2], f[3]});
        } else if (!m.legacy) {
            m.extra.push_back(line);
        }
    }
    for (auto& o : owners)  // owner lines name an action line by its action and path
        for (auto& e : m.e) if (e.owner.empty() && e.action == o[0] && IEq(e.rel, o[1])) { e.owner = o[2]; break; }
    // rc6 L1: the record of a game folder that was moved names the old folder; read from <dir>\wronggpu-setup\install.record
    // it is the record of dir (its entries are relative to it, its backup is next to it): rebased to dir
    if (header && !m.game.empty() && !m.legacy && IEq(BaseName(path), L"install.record") && IEq(BaseName(Parent(path)), kStateDirName))
        RebaseRecord(m, Parent(Parent(path)));
    return header && !m.game.empty();
}
// Always written as v2 (the launcher's format; a 0.5.x uninstaller reads it too, see record.h).
void SaveRecord(const wstring& path, const Record& m) {
    wstring t = L"WRONGGPU-RECORD 2\r\nversion\t" + m.version + L"\r\ngame\t" + m.game + L"\r\nexe\t" + m.exe + L"\r\nstatus\t" + m.status +
                L"\r\nregkey\t" + m.regkey + L"\r\nmodel\t" + m.model + L"\r\n";
    if (!m.modelM.empty()) t += L"model-m\t" + m.modelM + L"\r\n";
    auto dash = [](const wstring& s) { return s.empty() ? wstring(L"-") : s; };
    if (!m.payloadVersion.empty())
        t += L"payload\t" + m.payloadVersion + L"\t" + dash(m.runtimeSha) + L"\t" + dash(m.frontEndSha) + L"\t" + dash(m.pluginSha) +
             (m.rrPluginSha.empty() ? wstring() : L"\t" + m.rrPluginSha) + L"\r\n";
    if (!m.proxy.empty()) t += L"proxy\t" + m.proxy + L"\t" + m.proxyReason + L"\r\n";
    auto field = [](wstring s) { for (auto& c : s) if (c == L'\t' || c == L'\r' || c == L'\n') c = L' '; return s; };
    for (auto& c : m.coexist) t += L"coexist\t" + field(c.tool) + L"\t" + field(c.fingerprint) + L"\t" + field(c.decision) + L"\r\n";
    for (auto& c : m.inikeys)
        t += L"inikey\t" + field(c.file) + L"\t" + field(FromUtf8(c.section)) + L"\t" + field(FromUtf8(c.key)) + L"\t" +
             (c.absent ? wstring(L"absent") : L"=" + field(FromUtf8(c.original))) + L"\t=" + field(FromUtf8(c.ours)) + L"\r\n";
    if (const RrConfigFix& x = m.rrconfig; x.present) {
        wstring flags = wstring(x.created ? L"c" : L"") + (x.sectionAdded ? L"s" : L"") + (x.wasReadOnly ? L"w" : L"") + (x.setReadOnly ? L"r" : L"") + (x.backup ? L"b" : L"");
        t += L"rrconfig\t" + field(x.path) + L"\t" + (flags.empty() ? L"-" : flags) + L"\t" + (x.prevAbsent ? wstring(L"absent") : L"=" + field(FromUtf8(x.prev))) +
             L"\t" + (x.folder.empty() ? wstring(L"-") : field(x.folder)) + L"\r\n";
    }
    for (auto& x : m.extra) t += x + L"\r\n";
    for (auto& e : m.e)
        t += e.action + L"\t" + (e.existed ? L"1" : L"0") + L"\t" + (e.orig.empty() ? L"-" : e.orig) + L"\t" + (e.inst.empty() ? L"-" : e.inst) + L"\t" + e.rel + L"\r\n";
    for (auto& e : m.e)
        if (!e.owner.empty()) t += L"owner\t" + e.action + L"\t" + e.rel + L"\t" + e.owner + L"\r\n";
    string u = ToUtf8(t);
    EnsureDir(Parent(path));
    WriteFileAtomic(path, u.data(), u.size());
}

// Resolve a record-relative path below the install folder, refusing traversal and links.
wstring Target(const wstring& dir, const wstring& rel) {
    if (rel.empty() || rel.find(L':') != wstring::npos || rel[0] == L'\\' || rel[0] == L'/') Fail(L"Invalid path in the install record: " + rel);
    wstring cur = dir;
    size_t i = 0;
    while (i <= rel.size()) {
        size_t j = rel.find_first_of(L"\\/", i);
        if (j == wstring::npos) j = rel.size();
        wstring part = rel.substr(i, j - i);
        if (part == L"..") Fail(L"Invalid path in the install record: " + rel);
        if (!part.empty()) {
            cur = Join(cur, part);
            DWORD a = GetFileAttributesW(cur.c_str());
            if (a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_REPARSE_POINT)) Fail(L"Refusing to follow a link inside the game folder: " + cur);
        }
        i = j + 1;
    }
    return cur;
}

// A record of this route: its OptiScaler was put there by this installer, and the logs that OptiScaler writes while the
// game runs belong to the install (removed with it, never "another mod").
bool HostedRecord(const Record& m) { for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, L"OptiScaler.ini")) return true; return false; }

// ------------------------------------------------------------------------------------------------
// Uninstall registration (one entry per game)
// ------------------------------------------------------------------------------------------------
// The registry hive of the Apps & features entries: the player's. HKEY_CURRENT_USER, except in the elevated worker run
// by another account (an administrator's credentials typed at the UAC prompt): the player's hive under HKEY_USERS.
static HKEY g_uninstallHive = nullptr;
HKEY UninstallHive() { return g_uninstallHive ? g_uninstallHive : HKEY_CURRENT_USER; }
void SetUninstallHive(HKEY hive) { g_uninstallHive = hive; }
wstring RegKeyFor(const wstring& dir, const wchar_t* prefix) {
    uint32_t h = 2166136261u;
    for (wchar_t c : Lower(FullPath(dir))) { h ^= (uint32_t)c; h *= 16777619u; }
    wchar_t b[48]; swprintf(b, 48, L"%ls%08X", prefix, (unsigned)h);
    return b;
}
void RegisterUninstall(Context& ctx, const Record& m, const wstring& state, const wstring& displayName) {
    HKEY k;
    wstring key = UninstallRoot() + m.regkey;
    if (RegCreateKeyExW(UninstallHive(), key.c_str(), 0, nullptr, 0, KEY_WRITE, nullptr, &k, nullptr) != ERROR_SUCCESS) Fail(L"Cannot create the Apps & features entry.");
    auto setS = [&](const wchar_t* n, const wstring& v) { RegSetValueExW(k, n, 0, REG_SZ, (const BYTE*)v.c_str(), (DWORD)((v.size() + 1) * 2)); };
    auto setD = [&](const wchar_t* n, DWORD v) { RegSetValueExW(k, n, 0, REG_DWORD, (const BYTE*)&v, 4); };
    wstring exe = Join(state, L"uninstall.exe");
    setS(L"DisplayName", wstring(kAppName) + L" - " + displayName);
    setS(L"DisplayVersion", ctx.data->version);
    setS(L"Publisher", L"WrongGPU");
    setS(L"InstallLocation", m.game);
    setS(L"UninstallString", L"\"" + exe + L"\"");
    setS(L"QuietUninstallString", L"\"" + exe + L"\" /S");
    setS(L"ModifyPath", L"\"" + exe + L"\" /COLLECT");  // "Modify" in Apps & features = collect logs for a report
    setS(L"DisplayIcon", exe);
    setD(L"NoModify", 0); setD(L"NoRepair", 1);
    setD(L"EstimatedSize", (DWORD)(PayloadBytes(ctx) / 1024));
    RegCloseKey(k);
}
std::vector<wstring> RegisteredInstalls() {  // InstallLocation of every WrongGPU entry
    std::vector<wstring> r;
    HKEY root;
    if (RegOpenKeyExW(UninstallHive(), UninstallRoot().c_str(), 0, KEY_READ, &root) != ERROR_SUCCESS) return r;
    wchar_t sub[256];
    for (DWORD i = 0;; ++i) {
        DWORD n = (DWORD)std::size(sub);
        if (RegEnumKeyExW(root, i, sub, &n, nullptr, nullptr, nullptr, nullptr) != ERROR_SUCCESS) break;
        if (!StartsWithI(sub, kRegPrefix)) continue;
        r.push_back(RegString(UninstallHive(), (UninstallRoot() + sub).c_str(), L"InstallLocation"));
    }
    RegCloseKey(root);
    return r;
}
bool DriveUnreachable(const wstring& dir) {
    const wstring full = FullPath(dir);
    wstring root;
    if (full.size() >= 3 && full[1] == L':' && full[2] == L'\\') root = full.substr(0, 3);
    else if (full.rfind(L"\\\\", 0) == 0) {  // \\server\share\...
        const size_t a = full.find(L'\\', 2), b = a == wstring::npos ? a : full.find(L'\\', a + 1);
        if (a == wstring::npos) return false;
        root = (b == wstring::npos ? full : full.substr(0, b)) + L"\\";
    } else return false;
    return GetFileAttributesW(root.c_str()) == INVALID_FILE_ATTRIBUTES;
}
// A game on a drive that is not connected now (an unplugged USB drive, an offline share) still has WrongGPU and still uses
// the shared data once the drive is back: it counts, though its record cannot be read
wstring OtherGameWithWrongGPU(const wstring& dir) {
    auto counts = [&dir](const wstring& loc) {
        return !loc.empty() && (dir.empty() || !IEq(FullPath(loc), FullPath(dir))) && (IsFile(RecordPath(loc)) || DriveUnreachable(loc));
    };
    for (auto& loc : RegisteredInstalls()) if (counts(loc)) return loc;
    for (auto& loc : ManagedGames()) if (counts(loc)) return loc;
    return L"";
}
static wstring PortableListPath() {
    const wstring base = EnvVar(L"LOCALAPPDATA");
    return base.empty() ? wstring() : Join(Join(base, L"WrongGPU"), L"portable-launchers.txt");
}
static std::vector<wstring> PortableLaunchers() {
    std::vector<wstring> r;
    string raw;
    const wstring p = PortableListPath();
    if (p.empty() || !ReadAll(p, raw)) return r;
    const wstring text = FromUtf8(raw);
    for (size_t a = 0; a < text.size();) {
        size_t e = text.find(L'\n', a);
        if (e == wstring::npos) e = text.size();
        wstring l = text.substr(a, e - a);
        a = e + 1;
        while (!l.empty() && (l.back() == L'\r' || l.back() == L' ')) l.pop_back();
        if (!l.empty() && l[0] != L';') r.push_back(l);
    }
    return r;
}
static bool PortableLauncherThere(const wstring& exe) { return IsFile(exe) && IsFile(Join(Parent(exe), kPortableMarker)); }
void NotePortableLauncher(const wstring& exe) {
    static std::mutex noting;  // the window's workers (its health check, a conversion) may note at the same moment
    std::lock_guard<std::mutex> lock(noting);
    const wstring p = PortableListPath();
    if (p.empty() || exe.empty() || !IsDir(Parent(p))) return;
    std::vector<wstring> list = PortableLaunchers();
    for (auto& l : list) if (IEq(FullPath(l), FullPath(exe))) return;
    list.erase(std::remove_if(list.begin(), list.end(), [](const wstring& l) { return !PortableLauncherThere(l); }), list.end());  // copies deleted since
    list.push_back(FullPath(exe));
    wstring t = L"; WrongGPU: the portable launchers that use this data (one per line). Uninstallers read it.\r\n";
    for (auto& l : list) t += l + L"\r\n";
    const string u = ToUtf8(t);
    try { WriteFileAtomic(p, u.data(), u.size()); } catch (const SetupError&) {}
}
bool PortableLauncherUsesSharedData() {
    for (auto& l : PortableLaunchers()) if (PortableLauncherThere(l)) return true;
    return false;
}
bool LauncherKeepsSharedData() {
    if (LauncherInstalled()) return true;
    const wstring base = EnvVar(L"LOCALAPPDATA");
    if (!base.empty() && IsFile(Join(Join(base, L"WrongGPU"), L"launcher.ini"))) return true;
    return PortableLauncherUsesSharedData();
}
void UnregisterUninstall(const Record& m) {
    wstring key = m.regkey.empty() ? RegKeyFor(m.game) : m.regkey;
    RegDeleteKeyW(UninstallHive(), (UninstallRoot() + key).c_str());
}

// Files WrongGPU writes while the game runs are named wronggpu-* (README contract with the front end and runtime).
// Removed on uninstall from the install folder, the exe folder and the game root, except files in the record; with
// settings also the player's wronggpu.ini (a full uninstall; a re-install keeps it).
size_t RemoveRuntimeFiles(const wstring& dir, const wstring& exeDir, const Record& m, bool settings) {
    std::set<wstring> recorded;
    const bool opti = HostedRecord(m);
    for (auto& e : m.e) recorded.insert(Lower(FullPath(Join(dir, e.rel))));
    std::set<wstring> dirs{Lower(FullPath(dir))};
    if (!exeDir.empty()) dirs.insert(Lower(FullPath(exeDir)));
    // folders the install created (the plugin's own folder next to Ultimate ASI Loader keeps the plugin's wronggpu-* files)
    for (auto& e : m.e) if (e.action == L"dir") dirs.insert(Lower(FullPath(Join(dir, e.rel))));
    dirs.insert(Lower(FullPath(GameRoot(exeDir.empty() ? dir : exeDir))));
    // REFramework's mirror next to the exe (gamerules.h): the game runs WrongGPU's DLLs from there, so their logs land there
    if (const wstring mirror = Join(exeDir.empty() ? dir : exeDir, kMirrorDir); IsDir(mirror)) dirs.insert(Lower(FullPath(mirror)));
    // OptiScaler 10: the front end writes its wronggpu-* files next to itself in OptiScaler's custom upscaler slot
    if (const wstring s = Join(dir, kOptiSlotDir); IsDir(s)) dirs.insert(Lower(FullPath(s)));
    size_t removed = 0;
    for (auto& d : dirs) {
        for (auto& f : ListFiles(d, L"wronggpu-*"))
            if (!recorded.count(Lower(FullPath(f))) && DeleteFileW(f.c_str())) ++removed;
        if (settings && DeleteFileW(Join(d, L"wronggpu.ini").c_str())) ++removed;
        if (opti)  // logs written by the OptiScaler this installer set up
            for (auto pat : {L"OptiScaler.log", L"OptiScaler.previous*.log", L"fakenvapi.log"})
                for (auto& f : ListFiles(d, pat)) if (!recorded.count(Lower(FullPath(f))) && DeleteFileW(f.c_str())) ++removed;
    }
    return removed;
}

// ------------------------------------------------------------------------------------------------
// Restore / uninstall engine (from the 0.2.0 installer, extended with the rename and dir actions)
// ------------------------------------------------------------------------------------------------
// Restores every backed-up original and removes what the installer added.
// failedInstall: lenient mode used to roll back a half-finished install.
// keepState:     leave the state folder in place (re-install path); only the record + backup are cleared.
// Returns warnings (files left alone because they changed since install, e.g. by a game update).
// The running uninstaller cannot delete its own file (Windows refuses while the program runs, and also briefly after a
// temporary copy exits). The last step (its exe and the then empty wronggpu-setup folder) goes to a short-lived
// cmd.exe that waits for the uninstaller to exit. Started with CREATE_NO_WINDOW and no DETACHED_PROCESS: cmd and the
// programs it starts share one invisible console, so nothing appears on screen.
// state / self: the wronggpu-setup folder and the running uninstaller in it (empty: not this case); appData:
// %LOCALAPPDATA%\WrongGPU, removed with everything in it (empty: other games still use it).
void StartCleanup(Context& ctx, const wstring& state, const wstring& self, const wstring& appData) {
    // Detach this generation now. Delayed cleanup must never recursively delete the shared pathname,
    // which a following install can already have recreated with its new model and runtime selection.
    wstring retired;
    if (!appData.empty() && IsDir(appData)) {
        retired = appData + L".cleanup-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64());
        if (!MoveFileExW(appData.c_str(), retired.c_str(), MOVEFILE_WRITE_THROUGH)) {
            LogLine(ctx, L"Shared data cleanup deferred: cannot detach " + appData + L"; it was left in place.");
            retired.clear();
        } else if (StartsWithI(ctx.logFile, appData + L"\\")) {
            ctx.logFile = retired + ctx.logFile.substr(appData.size());
        }
    }
    wchar_t sys[MAX_PATH]; GetSystemDirectoryW(sys, MAX_PATH);
    const wstring cmdExe = Join(sys, L"cmd.exe"), pingExe = Join(sys, L"PING.EXE");
    wstring steps = L"\"" + pingExe + L"\" -n 3 127.0.0.1 >nul";
    if (!self.empty()) steps += L" & del /f /q \"" + self + L"\" & rmdir \"" + state + L"\"";
    if (!retired.empty()) steps += L" & rmdir /s /q \"" + retired + L"\"";
    wstring cmd = L"\"" + cmdExe + L"\" /D /C \"" + steps + L"\"";
    std::vector<wchar_t> buf(cmd.begin(), cmd.end()); buf.push_back(0);
    STARTUPINFOW si{}; si.cb = sizeof(si); si.dwFlags = STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};
    if (CreateProcessW(cmdExe.c_str(), buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, sys, &si, &pi)) {
        CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
        LogLine(ctx, L"final cleanup (" + (self.empty() ? wstring() : state + L" ") + appData + L") handed to a hidden cmd.exe");
    } else {
        LogLine(ctx, L"cleanup not started (error " + std::to_wstring(GetLastError()) + L"); remove " + state + L" by hand");
    }
}

std::vector<wstring> DoRestore(Context& ctx, const wstring& dir, Ui& ui, bool failedInstall, bool keepState) {
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy) Fail(L"No WrongGPU installation record was found in " + dir + L".", EXIT_NOT_INSTALLED);
    if (!IEq(m.game, dir)) Fail(L"The installation record belongs to a different folder: " + m.game);
    std::vector<wstring> warnings;
    std::set<size_t> skip;
    bool keepBackup = false;
    int total = (int)m.e.size() + 2, done = 0;
    // The FSR DLL an entry is about: WrongGPU's under the game's name (copy), the game's under its renamed name (rename).
    auto slotOf = [](const Entry& e) -> const FfxSlot* {
        for (auto& sl : kSlots) if (IEq(e.rel, e.action == L"rename" ? sl.original : sl.name)) return &sl;
        return nullptr;
    };
    std::map<size_t, wstring> from;  // a game FSR DLL that is restored from its renamed copy (no usable backup)

    if (!failedInstall) {
        ui.step(done, total, L"Checking the backup");
        for (size_t i = 0; i < m.e.size(); ++i) {
            auto& e = m.e[i];
            const FfxSlot* sl = e.action == L"copy" ? slotOf(e) : nullptr;
            if (e.existed) {
                wstring b = BackupPath(dir, e.rel);
                if (!Exists(b) || Sha256File(b) != e.orig) {
                    // The game's FSR DLL is in the folder a second time, next to WrongGPU's: that copy will do.
                    const wstring kept = sl ? Target(dir, sl->original) : wstring();
                    if (kept.empty() || Sha256File(kept) != e.orig) Fail(L"The backup of " + e.rel + L" is missing or damaged. Nothing was changed.");
                    from[i] = kept;
                }
            }
            if (e.action == L"copy" || e.action == L"rename") {
                wstring t = Target(dir, e.rel);
                if (Exists(t)) {
                    wstring h = Sha256File(t), info;
                    if (h != e.inst && h != e.orig) {
                        // the RR plugin's settings files: it rewrites them itself (its keys, the mode chosen in OptiScaler's
                        // menu); they are WrongGPU's all the same and go with the install
                        if (e.action == L"copy" && !e.existed && IsRrSettingsFile(e.rel)) {
                            ui.log(e.rel + L" was rewritten by WrongGPU's Ray Reconstruction plugin: removed with the rest");
                            continue;
                        }
                        // rc6 L5: the OptiScaler.ini of this install's OptiScaler setup, saved by OptiScaler's own menu: the
                        // install's all the same (removed with it; a player's file it replaced comes back from the backup)
                        if (e.action == L"copy" && IEq(e.rel, L"OptiScaler.ini")) {
                            ui.log(L"OptiScaler.ini was saved by OptiScaler's menu since the install: " +
                                   wstring(e.existed ? L"the file that was there before is put back" : L"removed with the rest"));
                            continue;
                        }
                        // Changed since install, but WrongGPU's own all the same (a newer build placed by hand): an FSR
                        // DLL by its version resource, the rest of the payload by its name (the settings file is the
                        // player's). It goes like the file the record names, and the game's original comes back.
                        if (e.action == L"copy" && (sl ? ClassifySlot(*sl, t, nullptr, info) == FFX_OURS
                                                       : !EndsWithI(e.rel, L".ini") && ProvablyOurs(ctx, t))) {
                            ui.log(e.rel + L" is not the file that was installed, but it is WrongGPU's own" + (sl ? L" (" + info + L")" : wstring()) + L": removed with the rest");
                            continue;
                        }
                        skip.insert(i);
                        warnings.push_back(e.action == L"copy" && IsSlotName(e.rel) ? L"The game replaced " + e.rel + L" (probably an update); its new file was kept."
                                                               : L"Left in place (changed since install): " + e.rel);
                    }
                }
            }
        }
    }
    for (size_t k = m.e.size(); k-- > 0;) {  // reverse order of installation
        auto& e = m.e[k];
        ui.step(++done, total, L"Restoring " + e.rel);
        if (skip.count(k)) { ui.log(L"kept " + e.rel); continue; }
        wstring t = Target(dir, e.rel), b = BackupPath(dir, e.rel);
        if (e.action == L"dir") {
            if (IsDir(t) && RemoveDirectoryW(t.c_str())) ui.log(L"removed folder " + e.rel);
        } else if (e.action == L"copy" || e.action == L"rename") {
            // WrongGPU's DLL is still under the game's name (no entry or no backup for it): the renamed file is the
            // game's only copy. It goes back under its name instead of being removed.
            const FfxSlot* sl = e.action == L"rename" ? slotOf(e) : nullptr;
            wstring info;
            if (sl && Exists(t) && (e.inst.empty() || Sha256File(t) == e.inst) && ClassifySlot(*sl, Join(dir, sl->name), nullptr, info) == FFX_OURS) {
                SetFileAttributesW(Join(dir, sl->name).c_str(), FILE_ATTRIBUTE_NORMAL);
                if (!MoveFileExW(t.c_str(), Join(dir, sl->name).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                    Fail(L"Cannot put " + wstring(sl->name) + L" back: " + ErrText(GetLastError()));
                ui.log(L"restored " + wstring(sl->name) + L" from " + e.rel);
            }
            if (e.existed) {
                if (failedInstall && !Exists(b)) continue;
                const auto kept = from.find(k);
                CopyChecked(kept == from.end() ? b : kept->second, t);
                if (Sha256File(t) != e.orig) Fail(L"Restore verification failed for " + e.rel);
                ui.log(L"restored " + e.rel);
            } else if (e.owner == L"adopted") {
                ui.log(L"kept " + e.rel + L" (the player's file, used by WrongGPU but never deleted)");
            } else if (Exists(t)) { ForceDelete(t); ui.log(L"removed " + e.rel); }
        } else {  // disable: put the original file back
            if (!Exists(b)) { if (failedInstall) continue; Fail(L"Backup of " + e.rel + L" is missing."); }
            if (Exists(t)) {
                if (Sha256File(t) != e.orig) { keepBackup = true; warnings.push_back(e.rel + L" exists again with different content; the original stays in the backup folder."); }
            } else { CopyChecked(b, t); ui.log(L"restored " + e.rel); }
        }
    }
    // REFramework's copies of the install's root DLLs that the install did not write itself (REFramework came, or made its
    // mirror, after the install; an install from before this build): removed when byte for byte the file installed, so
    // the game cannot go on loading WrongGPU's files from there. Anything else there stays (gamerules.h).
    if (IEq(FullPath(dir), FullPath(m.exe.empty() ? dir : Parent(m.exe))) && IsDir(Join(dir, kMirrorDir))) {
        std::set<wstring> tracked;
        for (auto& e : m.e) if (IsMirrorRel(e.rel)) tracked.insert(Lower(e.rel));
        for (size_t k = 0; k < m.e.size(); ++k) {
            const Entry& e = m.e[k];
            if (e.action != L"copy" || !MirroredRel(e.rel) || e.inst.empty() || skip.count(k) || tracked.count(Lower(MirrorRel(e.rel)))) continue;
            const wstring t = Target(dir, MirrorRel(e.rel));
            if (IsFile(t) && Sha256File(t) == e.inst) {
                ForceDelete(t);
                ui.log(L"removed " + MirrorRel(e.rel) + L" (REFramework's copy of WrongGPU's " + e.rel + L")");
            }
        }
    }
    ui.step(++done, total, L"Cleaning up");
    if (!failedInstall) {
        if (size_t n = RemoveRuntimeFiles(dir, m.exe.empty() ? L"" : Parent(m.exe), m, !keepState))
            ui.log(L"removed " + std::to_wstring(n) + L" settings, log and report file(s) written by WrongGPU");
        for (size_t k = m.e.size(); k-- > 0;)  // a folder the install made that held only those files
            if (m.e[k].action == L"dir" && !skip.count(k) && IsDir(Target(dir, m.e[k].rel)) && RemoveDirectoryW(Target(dir, m.e[k].rel).c_str()))
                ui.log(L"removed folder " + m.e[k].rel);
    }
    // the game's Engine.ini outside the game folder (rrconfig.h, rr_config_fix=on): its line taken out again
    if (!failedInstall && !keepState && m.rrconfig.present) {
        std::vector<wstring> notes;
        RevertRrConfig(ctx, dir, m, notes, warnings);
        for (auto& n : notes) ui.log(n);
    }
    UnregisterUninstall(m);
    // The model data and everything else in %LOCALAPPDATA%\WrongGPU are shared by all games: they go when the last game
    // is uninstalled (the folder itself at the very end, by the hidden cleanup, so later log lines cannot recreate it).
    // A launcher that uses them keeps them (it shows the model and RR data as its own; its uninstall removes them).
    wstring appData;
    if (!keepState && !failedInstall) {
        // The launcher's games (one Apps & features entry for all, no entry per game): its list of managed folders.
        RemoveManagedGame(dir);
        // a moved game folder (rc6 L1): the folder the record named before, when nothing of WrongGPU is left there
        if (!m.movedFrom.empty() && !IsFile(RecordPath(m.movedFrom)) && !DriveUnreachable(m.movedFrom)) RemoveManagedGame(m.movedFrom);
        const wstring other = OtherGameWithWrongGPU(dir);
        if (!other.empty()) LogLine(ctx, L"the shared model data stays: " + other + L" still uses it");
        else if (LauncherKeepsSharedData()) LogLine(ctx, L"the shared model data stays: the launcher uses it (its own uninstall removes it)");
        else {
            if (IsDir(ModelDir(ctx))) { RemoveTree(ModelDir(ctx)); ui.log(L"removed the model data (no other game uses it)"); }
            const wstring base = EnvVar(L"LOCALAPPDATA");
            if (!base.empty()) appData = Join(base, L"WrongGPU");
        }
    }

    wstring state = StateDir(dir);
    if (keepBackup) { warnings.push_back(L"The backup folder was kept: " + Join(state, L"backup")); if (!appData.empty()) StartCleanup(ctx, L"", L"", appData); return warnings; }
    if (keepState) { RemoveTree(Join(state, L"backup")); DeleteFileW(RecordPath(dir).c_str()); return warnings; }
    wstring self = SelfPath();
    if (IEq(Parent(self), state)) {
        RemoveTree(state, self);
        StartCleanup(ctx, state, self, appData);
    } else {
        RemoveTree(state);
        if (!appData.empty()) StartCleanup(ctx, L"", L"", appData);
    }
    return warnings;
}

static bool SettingsName(const wstring& name, bool folder) {
    return folder ? IEq(name, L"previous") || IEq(name, L"rr-config") : IEq(name, L"profile.ini");
}
bool StateHasSettings(const wstring& dir) {
    const wstring state = StateDir(dir);
    for (auto& f : ListFiles(state, L"*")) if (SettingsName(BaseName(f), false)) return true;
    for (auto& d : ListDirs(state)) if (SettingsName(BaseName(d), true)) return true;
    return false;
}
void RemoveStateKeepingSettings(const wstring& dir) {
    const wstring state = StateDir(dir);
    for (auto& f : ListFiles(state, L"*"))  // never throws (a destructor calls it)
        if (!SettingsName(BaseName(f), false)) { SetFileAttributesW(f.c_str(), FILE_ATTRIBUTE_NORMAL); DeleteFileW(f.c_str()); }
    for (auto& d : ListDirs(state)) if (!SettingsName(BaseName(d), true)) RemoveTree(d);
    RemoveDirectoryW(state.c_str());  // only when empty
}

// Byte-exact removal of a WrongGPU 0.1.x/0.2.x ("DLSS 4.5 AMD") install from its own record and backup: the same
// rules as that version's uninstaller, without running its program.
std::vector<wstring> LegacyRestore(Context& ctx, const wstring& dir, Ui& ui) {
    Record m;
    if (!LoadRecord(LegacyRecordPath(dir), m) || !m.legacy) Fail(L"The older install's record in " + dir + L" cannot be read.");
    std::vector<wstring> warnings;
    std::set<size_t> skip;
    for (size_t i = 0; i < m.e.size(); ++i) {
        auto& e = m.e[i];
        if (e.existed) {
            wstring b = LegacyBackupPath(dir, e.rel);
            if (!Exists(b) || Sha256File(b) != e.orig) Fail(L"The backup of the older install (" + e.rel + L") is missing or damaged. Uninstall it with its own uninstaller first.", EXIT_OLD_INSTALL);
        }
        if (e.action == L"copy") {
            wstring t = Target(dir, e.rel);
            if (Exists(t)) { wstring h = Sha256File(t); if (h != e.inst && h != e.orig) { skip.insert(i); warnings.push_back(L"Older install: left in place (changed since install): " + e.rel); } }
        }
    }
    for (size_t i = 0; i < m.e.size(); ++i) {
        auto& e = m.e[i];
        if (skip.count(i)) continue;
        wstring t = Target(dir, e.rel), b = LegacyBackupPath(dir, e.rel);
        if (e.action == L"copy") {
            if (e.existed) { CopyChecked(b, t); if (Sha256File(t) != e.orig) Fail(L"Restore verification failed for " + e.rel); ui.log(L"older install: restored " + e.rel); }
            else if (Exists(t)) { ForceDelete(t); ui.log(L"older install: removed " + e.rel); }
        } else if (!Exists(t)) { CopyChecked(b, t); ui.log(L"older install: restored " + e.rel); }
    }
    for (const wchar_t* d : {L"dlss45-model\\gpu-build", L"dlss45-model\\exposure-build", L"dlss45-model"}) RemoveDirectoryW(Join(dir, d).c_str());
    std::set<wstring> recorded;
    for (auto& e : m.e) recorded.insert(Lower(FullPath(Target(dir, e.rel))));
    for (const wstring& d : {dir, GameRoot(dir)})
        for (auto pat : {L"dlss45-*.log", L"dlss45-*.log.old", L"dlss45-report.txt", L"dlss45-report.json", L"dlss45-dred.txt", L"dlss45-crash-*.txt",
                         L"dlss45-crash-*.dmp", L"dlss45-capture-*.bin", L"dlss45-capture-*.txt", L"dlss45-capture-*.jpg"})
            for (auto& f : ListFiles(d, pat)) if (!recorded.count(Lower(FullPath(f)))) DeleteFileW(f.c_str());
    wstring key = m.regkey.empty() ? RegKeyFor(dir, kLegacyKeyPrefix) : m.regkey;
    RegDeleteKeyW(UninstallHive(), (UninstallRoot() + key).c_str());
    wstring legacy = UninstallRoot() + kLegacyKey;
    if (IEq(RegString(UninstallHive(), legacy.c_str(), L"InstallLocation"), dir)) RegDeleteKeyW(UninstallHive(), legacy.c_str());
    RemoveTree(LegacyStateDir(dir));
    ui.log(L"the older WrongGPU (DLSS 4.5 AMD) install in " + dir + L" was removed");
    LogLine(ctx, L"legacy install removed from " + dir);
    return warnings;
}

// ------------------------------------------------------------------------------------------------
// Record v2 (the launcher)
// ------------------------------------------------------------------------------------------------
// The launcher's list of the game folders it manages, in the shared data folder (%LOCALAPPDATA%\WrongGPU): read by
// every uninstaller, so the model data stays while another launcher-managed game uses it.
static wstring ManagedListPath() {
    const wstring base = EnvVar(L"LOCALAPPDATA");
    return base.empty() ? wstring() : Join(Join(base, L"WrongGPU"), L"managed-games.txt");
}
std::vector<wstring> ManagedGames() {
    std::vector<wstring> r;
    string raw;
    const wstring p = ManagedListPath();
    if (p.empty() || !ReadAll(p, raw)) return r;
    const wstring text = FromUtf8(raw);
    for (size_t a = 0; a < text.size();) {
        size_t e = text.find(L'\n', a);
        if (e == wstring::npos) e = text.size();
        wstring l = text.substr(a, e - a);
        a = e + 1;
        while (!l.empty() && (l.back() == L'\r' || l.back() == L' ')) l.pop_back();
        if (!l.empty() && l[0] != L';') r.push_back(l);
    }
    return r;
}
static void WriteManaged(const std::vector<wstring>& list) {
    const wstring p = ManagedListPath();
    if (p.empty()) return;
    wstring t = L"; WrongGPU launcher: the game folders it manages (one per line). Uninstallers read it.\r\n";
    for (auto& l : list) t += l + L"\r\n";
    const string u = ToUtf8(t);
    EnsureDir(Parent(p));
    WriteFileAtomic(p, u.data(), u.size());
}
void AddManagedGame(const wstring& dir) {
    auto list = ManagedGames();
    for (auto& l : list) if (IEq(FullPath(l), FullPath(dir))) return;
    list.push_back(FullPath(dir));
    WriteManaged(list);
}
void RemoveManagedGame(const wstring& dir) {
    auto list = ManagedGames();
    const size_t n = list.size();
    list.erase(std::remove_if(list.begin(), list.end(), [&](const wstring& l) { return IEq(FullPath(l), FullPath(dir)); }), list.end());
    if (list.size() != n) WriteManaged(list);
}

// Files of WrongGPU's payload by name: the payload's own paths, WrongGPU's FFX DLL under its hosted name, the
// plugin, and anything named wronggpu*.
static bool IsOurFile(Context& ctx, const wstring& rel) {
    if (StartsWithI(BaseName(rel), L"wronggpu") || IsOptiFrontEnd(rel) || IEq(rel, kAsiPlugin)) return true;
    if (ctx.data) for (auto& f : *ctx.data) if (IEq(f.rel, rel)) return true;
    return false;
}
void InferOwners(Context& ctx, Record& m) {
    const bool hosted = HostedRecord(m);
    for (auto& e : m.e) {
        if (!e.owner.empty()) continue;
        if (e.action == L"disable" || e.action == L"rename") e.owner = L"moved";
        else if (e.action == L"dir") e.owner = L"wg";
        else if (hosted && !IsOurFile(ctx, e.rel) && !IsSlotName(e.rel)) e.owner = L"wg-optibundle";
        else e.owner = L"wg";
    }
}
bool RecordCopies(const Record& m, const wstring& rel) {
    for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, rel)) return true;
    return false;
}
void SetPayloadLine(Context& ctx, Record& m) {
    m.payloadVersion = ctx.data ? ctx.data->version : L"";
    m.runtimeSha.clear(); m.frontEndSha.clear(); m.pluginSha.clear(); m.rrPluginSha.clear();
    if (!ctx.data) return;
    for (auto& f : *ctx.data) {
        if (IEq(f.rel, L"wronggpu_runtime.dll")) m.runtimeSha = f.sha256;
        else if (IEq(f.rel, L"amd_fidelityfx_dx12.dll")) m.frontEndSha = f.sha256;
        else if (IEq(f.rel, kAsiPlugin) && RecordCopies(m, kAsiPlugin)) m.pluginSha = f.sha256;  // the slot route has no menu plugin
    }
    // the RR plugin: the build DoInstall picks for this game (by its game interface version), only where the record has RR files
    if (RecordHasRr(m))
        if (const PayloadFile* rp = RrPluginFor(ctx, GameInterfaceVersion(ctx, m.exe))) m.rrPluginSha = rp->sha256;
}
// The RR plugin build an install placed: the payload line's sixth field, else (a record before rc6) its RR plugin entry's hash.
static wstring InstalledRrPluginSha(const Record& m) {
    if (!m.rrPluginSha.empty()) return m.rrPluginSha;
    for (auto& e : m.e) if (e.action == L"copy" && IsRrPlugin(e.rel)) return e.inst;
    return L"";
}
bool PayloadDiffers(Context& ctx, const Record& m) {
    if (!ctx.data) return false;
    if (m.payloadVersion.empty()) return true;  // no payload line: another build (one that didn't write it)
    Record cur = m;                               // the same entries: the menu plugin counts only where the record copies it
    SetPayloadLine(ctx, cur);
    return !IEq(m.runtimeSha, cur.runtimeSha) || !IEq(m.frontEndSha, cur.frontEndSha) || !IEq(m.pluginSha, cur.pluginSha) ||
           !IEq(InstalledRrPluginSha(m), cur.rrPluginSha);
}
string OwnedFilesJson(const Record& m) {
    Json j;
    j.Obj().S("format", "wronggpu-owned-files/1").S("note", L"Written by WrongGPU from its install record; read-only information for other tools.")
        .S("version", m.version).S("folder", m.game).S("exe", m.exe);
    j.Key("files").Arr();
    for (auto& e : m.e) {
        j.Obj().S("path", e.rel).S("action", e.action).S("owner", e.owner.empty() ? wstring(L"wg") : e.owner);
        if (!e.inst.empty()) j.S("sha256", e.inst);
        if (e.existed) j.S("original_sha256", e.orig);
        j.End();
    }
    j.End().End();
    return j.str() + "\n";
}
void WriteOwnedFiles(const wstring& dir, const Record& m) {
    const string s = OwnedFilesJson(m);
    WriteFileAtomic(Join(StateDir(dir), L"owned-files.json"), s.data(), s.size());
}
bool UpgradeRecord(Context& ctx, const wstring& dir, std::vector<wstring>* notes) {
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy) return false;
    if (m.format >= 2) return false;
    CopyChecked(RecordPath(dir), RecordPath(dir) + L".v1");
    InferOwners(ctx, m);
    if (m.payloadVersion.empty()) {  // what the 0.5.x install wrote: its version, the hashes from its own entries
        m.payloadVersion = m.version;
        for (auto& e : m.e) {
            if (e.action != L"copy") continue;
            if (IEq(e.rel, L"wronggpu_runtime.dll")) m.runtimeSha = e.inst;
            else if (IsOptiFrontEnd(e.rel) || IEq(e.rel, L"amd_fidelityfx_dx12.dll")) m.frontEndSha = e.inst;
            else if (IEq(e.rel, kAsiPlugin)) m.pluginSha = e.inst;
        }
    }
    if (m.proxy.empty())
        for (auto& e : m.e)
            if (e.owner == L"wg-optibundle")
                for (auto px : kOptiProxies) if (IEq(e.rel, px)) { m.proxy = px; m.proxyReason = L"chosen by setup " + m.version; }
    SaveRecord(RecordPath(dir), m);
    if (notes) notes->push_back(L"install record upgraded to v2 (the 0.5.x record is kept as install.record.v1)");
    return true;
}
bool SaveRebasedRecord(Context& ctx, const wstring& dir, Record& m) {
    if (m.movedFrom.empty()) return false;
    const wstring old = m.movedFrom;
    wstring from, to;
    MovedPrefix(old, dir, from, to);
    try {
        // OptiScaler.ini of this install: values that are paths into the old folder (the front end, the plugins folder)
        if (HostedRecord(m)) {
            const wstring ini = Join(dir, L"OptiScaler.ini");
            string text;
            if (!from.empty() && ReadAll(ini, text)) {
                const wstring before = Sha256Bytes(text.data(), text.size());
                string out;
                size_t changed = 0;
                for (size_t a = 0; a < text.size();) {
                    size_t e = text.find('\n', a);
                    e = e == string::npos ? text.size() : e + 1;
                    string line = text.substr(a, e - a);
                    a = e;
                    const size_t eq = line.find('=');
                    const size_t lead = line.find_first_not_of(" \t");
                    if (eq != string::npos && lead != string::npos && line[lead] != ';' && line[lead] != '#' && line[lead] != '[') {
                        size_t vb = line.find_first_not_of(" \t", eq + 1), ve = line.find_last_not_of(" \t\r\n");
                        if (vb != string::npos && ve != string::npos && ve >= vb) {
                            const wstring v = FromUtf8(line.substr(vb, ve - vb + 1)), r = Rebased(v, from, to);
                            if (r != v) { line = line.substr(0, vb) + ToUtf8(r) + line.substr(ve + 1); ++changed; }
                        }
                    }
                    out += line;
                }
                if (changed) {
                    WriteFileAtomic(ini, out.data(), out.size());
                    // the record's hash follows only a file that was as the launcher last wrote it (an outside change stays one)
                    for (auto& e : m.e)
                        if (e.action == L"copy" && IEq(e.rel, L"OptiScaler.ini") && IEq(e.inst, before)) e.inst = Sha256Bytes(out.data(), out.size());
                    LogLine(ctx, L"moved game: " + std::to_wstring(changed) + L" OptiScaler.ini path(s) now point into " + dir);
                }
            }
        }
        m.movedFrom.clear();
        SaveRecord(RecordPath(dir), m);
        WriteOwnedFiles(dir, m);
    } catch (const SetupError& e) {
        m.movedFrom = old;
        LogLine(ctx, L"moved game: the record in " + dir + L" names " + old + L"; not written again (" + e.msg + L"), it is read as this folder's");
        return false;
    }
    LogLine(ctx, L"moved game: the record in " + dir + L" named " + old + L"; its paths now name " + dir);
    // the game's own Apps & features entry (an install without the launcher's entry): its uninstaller is in the new folder
    if (!m.regkey.empty()) {
        const wstring key = UninstallRoot() + m.regkey;
        if (IEq(FullPath(RegString(UninstallHive(), key.c_str(), L"InstallLocation")), FullPath(old))) {
            HKEY k;
            if (RegOpenKeyExW(UninstallHive(), key.c_str(), 0, KEY_SET_VALUE, &k) == ERROR_SUCCESS) {
                const wstring un = Join(StateDir(dir), L"uninstall.exe");
                auto setS = [&](const wchar_t* n, const wstring& v) { RegSetValueExW(k, n, 0, REG_SZ, (const BYTE*)v.c_str(), (DWORD)((v.size() + 1) * 2)); };
                setS(L"InstallLocation", dir);
                setS(L"UninstallString", L"\"" + un + L"\"");
                setS(L"QuietUninstallString", L"\"" + un + L"\" /S");
                setS(L"ModifyPath", L"\"" + un + L"\" /COLLECT");
                setS(L"DisplayIcon", un);
                RegCloseKey(k);
                LogLine(ctx, L"moved game: its Apps & features entry now names " + dir);
            }
        }
    }
    // the launcher's list: the new folder in the old one's place (kept while the old one's drive isn't connected). A record
    // without its own entry is the launcher's (the scan may have taken the old folder off the list already)
    try {
        bool listed = m.regkey.empty();
        for (auto& l : ManagedGames()) listed = listed || IEq(FullPath(l), FullPath(old));
        if (listed) {
            AddManagedGame(dir);
            if (!IsFile(RecordPath(old)) && !DriveUnreachable(old)) RemoveManagedGame(old);
        }
    } catch (const SetupError&) {}
    return true;
}

std::vector<wstring> VerifyInstall(const wstring& dir, const Record& m) {
    std::vector<wstring> problems;
    for (auto& e : m.e) {
        if ((e.action != L"copy" && e.action != L"rename") || e.inst.empty()) continue;
        if (IsMirrorRel(e.rel)) continue;  // REFramework's copies (gamerules.h): it writes them again itself
        const wstring t = Join(dir, e.rel);
        if (!IsFile(t)) { problems.push_back(L"missing: " + e.rel); continue; }
        if (EndsWithI(e.rel, L".ini")) continue;  // settings: changed by the game, OptiScaler and the launcher
        if (Sha256File(t) != e.inst) problems.push_back(L"changed: " + e.rel);
    }
    return problems;
}

}  // namespace wg