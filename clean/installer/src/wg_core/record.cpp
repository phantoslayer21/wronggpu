// SPDX-License-Identifier: MIT
// WrongGPU launcher core: the install record (v1), the "Apps & features" entry and the restore / uninstall engine.
// Written for this project.
#include "record.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cwchar>
#include <random>
#ifdef WRONGGPU_TESTING
#include <restartmanager.h>
#pragma comment(lib, "rstrtmgr.lib")
#endif

#include "detect.h"
#include "fg.h"
#include "fingerprint.h"
#include "gamerules.h"
#include "json_writer.h"
#include "launcher.h"
#include "leftovers.h"
#include "optiscaler.h"
#include "payload.h"
#include "rr.h"
#include "rrconfig.h"
#include "settings.h"
#include "../../../wgshared/include/wg_state.h"  // the game's state files (settings.ini, model-choice.ini) and the state lock

namespace wg {

wstring StateDir(const wstring& dir) { return Join(dir, kStateDirName); }
wstring RecordPath(const wstring& dir) { return Join(StateDir(dir), L"install.record"); }
wstring BackupPath(const wstring& dir, const wstring& rel) { return Join(Join(StateDir(dir), L"backup"), rel); }
wstring EntryBackupPath(const wstring& dir, const Entry& e) { return BackupPath(dir, e.backup.empty() ? e.rel : e.backup); }
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
    m = Record();  // the file's record alone (#219: a second load into the same Record listed every entry again)
    wstring text = FromUtf8(raw);
    size_t pos = 0; bool header = false;
    std::vector<std::array<wstring, 3>> owners, backups;
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
            else if (!m.legacy && f[0] == L"install-id") m.installId = f[1];
            else if (!m.legacy && f[0] == L"rrconfig-backup") m.rrconfig.backupSha = f[1];
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
        } else if (!m.legacy && f[0] == L"backup" && f.size() == 4 && f[3].find(L"..") == wstring::npos && f[3].find_first_of(L"/:") == wstring::npos) {
            backups.push_back({f[1], f[2], f[3]});
        } else if (!m.legacy && f[0] == L"runtime" && f.size() == 3 && StartsWithI(f[2], L"wronggpu-") && f[2].find_first_of(L"\\/:") == wstring::npos) {
            m.runtime.push_back({f[1], f[2]});
        } else if (!m.legacy) {
            m.extra.push_back(line);
        }
    }
    for (auto& o : owners)  // owner lines name an action line by its action and path
        for (auto& e : m.e) if (e.owner.empty() && e.action == o[0] && IEq(e.rel, o[1])) { e.owner = o[2]; break; }
    for (auto& o : backups)  // backup lines too: the entry's backup under another name
        for (auto& e : m.e) if (e.backup.empty() && e.action == o[0] && IEq(e.rel, o[1])) { e.backup = o[2]; break; }
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
    if (!m.installId.empty()) t += L"install-id\t" + m.installId + L"\r\n";
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
        if (x.backup && !x.backupSha.empty()) t += L"rrconfig-backup\t" + field(x.backupSha) + L"\r\n";
    }
    for (auto& r : m.runtime) t += L"runtime\t" + field(r.folder) + L"\t" + field(r.pattern) + L"\r\n";
    for (auto& x : m.extra) t += x + L"\r\n";
    for (auto& e : m.e)
        t += e.action + L"\t" + (e.existed ? L"1" : L"0") + L"\t" + (e.orig.empty() ? L"-" : e.orig) + L"\t" + (e.inst.empty() ? L"-" : e.inst) + L"\t" + e.rel + L"\r\n";
    for (auto& e : m.e)
        if (!e.owner.empty()) t += L"owner\t" + e.action + L"\t" + e.rel + L"\t" + e.owner + L"\r\n";
    for (auto& e : m.e)
        if (!e.backup.empty()) t += L"backup\t" + e.action + L"\t" + e.rel + L"\t" + e.backup + L"\r\n";
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
bool OptiIniWrittenByUs(const wstring& dir, const Record& m) {
    const wstring p = Join(dir, L"OptiScaler.ini");
    if (!IsFile(p)) return false;
    for (auto& e : m.e)
        if (e.action == L"copy" && IEq(e.rel, L"OptiScaler.ini")) return !e.inst.empty() && IEq(Sha256File(p), e.inst);
    return false;
}
bool OptiSetupIsOurs(const wstring& dir, const Record& m) {
    if (!IsFile(Join(dir, L"OptiScaler.ini"))) return false;
    if (OptiIniWrittenByUs(dir, m)) return true;
    if (!m.proxy.empty())
        for (auto& e : m.e)
            if (e.action == L"copy" && IEq(e.rel, m.proxy) && !e.inst.empty())
                return IsFile(Target(dir, e.rel)) && IEq(Sha256File(Target(dir, e.rel)), e.inst);
    return false;
}

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

// Files WrongGPU's own programs write while the game runs are named wronggpu-* (README contract with the front end and
// runtime). The install lists the folders they write to in its record (runtime lines, SetRuntimeLines): the install folder,
// the exe folder and REFramework's mirror when they are in it, OptiScaler 10's custom upscaler slot, and the folders the
// install created. A file there that the install did not record has no bytes to prove it WrongGPU's (a player can save
// wronggpu-notes.txt, another program can write a similar name): it is never deleted. The restore finds them
// (UnrecordedRuntimeFiles) and says so; wronggpu.ini and the other recorded files go by their hash like any other.
void SetRuntimeLines(Record& m, const wstring& dir, const wstring& exeDir) {
    m.runtime.clear();
    auto add = [&](const wstring& folder) {
        const wstring full = FullPath(folder), base = FullPath(dir);
        wstring rel;
        if (IEq(full, base)) rel = L".";
        else if (StartsWithI(full, base + L"\\")) rel = full.substr(base.size() + 1);
        else return;  // outside the install folder: never listed
        for (auto& r : m.runtime) if (IEq(r.folder, rel)) return;
        m.runtime.push_back({rel, L"wronggpu-*"});
    };
    add(dir);
    const wstring ed = exeDir.empty() ? dir : exeDir;
    add(ed);
    add(Join(ed, kMirrorDir));
    add(Join(dir, kOptiSlotDir));
    for (auto& e : m.e) if (e.action == L"dir") add(Join(dir, e.rel));
    // The game root (GameRoot of the exe folder, the folder Collect logs reads as "game-root"): the game's working folder,
    // where WrongGPU's programs write their wronggpu-* logs too. Above the install folder it is a named line, resolved at the
    // restore from the record's exe the same way.
    const wstring root = FullPath(GameRoot(FullPath(ed))), base = FullPath(dir);
    if (IEq(root, base) || StartsWithI(root, base + L"\\")) add(root);
    else if (std::none_of(m.runtime.begin(), m.runtime.end(), [](const auto& r) { return r.folder == kRuntimeGameRoot; }))
        m.runtime.push_back({kRuntimeGameRoot, L"wronggpu-*"});
}
std::vector<wstring> UnrecordedRuntimeFiles(const wstring& dir, const Record& m) {
    std::set<wstring> recorded, seen;
    for (auto& e : m.e) recorded.insert(Lower(FullPath(Join(dir, e.rel))));
    std::vector<wstring> found;
    for (auto& r : m.runtime) {
        wstring d;
        if (r.folder == kRuntimeGameRoot) d = GameRoot(m.exe.empty() ? dir : Parent(m.exe));
        else try { d = r.folder == L"." ? dir : Target(dir, r.folder); } catch (const SetupError&) { continue; }
        if (!IsDir(d)) continue;
        for (auto& f : ListFiles(d, r.pattern.c_str()))
            if (StartsWithI(BaseName(f), L"wronggpu-") && !recorded.count(Lower(FullPath(f))) && seen.insert(Lower(FullPath(f))).second) found.push_back(f);
    }
    return found;
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
// ---- the state folder (<game folder>\wronggpu-setup) ----------------------------------------------------------------------
// It is in the game's folder, so it keeps the own-files-only rule: a file a player saved there is not deleted. Only what WrongGPU
// wrote goes, with its bytes checked where a hash is known (StateOwned). Everything else stays, the folder stays with it, and
// the result lists what was left. (The shared data folder %LOCALAPPDATA%\WrongGPU is WrongGPU's own application folder: the last
// game's uninstall removes it whole, StartCleanup.)
static wstring LowerRel(wstring r) {
    for (auto& c : r) if (c == L'/') c = L'\\';
    return Lower(r);
}
// What the state folder holds that WrongGPU wrote, with the bytes it must have: its path below the folder (lower case) -> SHA-256
// ("": by name only). The backups the record lists (the hash is the entry's `orig`), the previous version's files (the hash
// previous.txt states) and owned-files.json (the bytes WrongGPU makes of this record) are deleted only while their bytes are
// those; the record, previous.txt, uninstall.exe, setup.log and profile.ini carry no recorded hash and go by their fixed names.
static std::function<bool(const wstring&, PinnedFile&)> StateOwned(const wstring& dir, const Record* m) {
    auto names = std::make_shared<std::map<wstring, wstring>>();
    for (const wchar_t* n : {L"install.record", L"uninstall.exe", L"setup.log", L"profile.ini"}) (*names)[n] = L"";
    // #219: the game's state files (wg_state.h: settings chosen in the launcher or a WrongGPU menu in the game, the model choice
    // record) are WrongGPU's only while they carry WrongGPU's stamp valid for THEIR OWN NAME and THIS INSTALL's id (the record's
    // install-id; wg_state.h StampedByWrongGPU). An edited file, a copy under another name (a temporary file among them) or
    // from another install, a file without the stamp: not WrongGPU's, it stays and is listed (below, in the returned test)
    const string installId = m ? ToUtf8(m->installId) : string();
    if (m) {
        const string json = OwnedFilesJson(*m);
        (*names)[L"owned-files.json"] = Sha256Bytes(json.data(), json.size());
        for (auto& e : m->e) if (e.action != L"dir" && !e.orig.empty()) (*names)[LowerRel(L"backup\\" + (e.backup.empty() ? e.rel : e.backup))] = Lower(e.orig);
    }
    string raw;
    if (ReadAll(Join(Join(StateDir(dir), L"previous"), L"previous.txt"), raw)) {  // the kept version: "file=<sha>|<rel>" lines
        (*names)[L"previous\\previous.txt"] = L"";
        for (size_t a = 0; a < raw.size();) {
            size_t e = raw.find('\n', a);
            if (e == string::npos) e = raw.size();
            string line = raw.substr(a, e - a);
            a = e + 1;
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            const size_t bar = line.find('|');
            if (line.rfind("file=", 0) == 0 && bar != string::npos)
                (*names)[LowerRel(L"previous\\" + FromUtf8(line.substr(bar + 1)))] = Lower(FromUtf8(line.substr(5, bar - 5)));
        }
    }
    return [names, installId](const wstring& rel, PinnedFile& f) {
        const wstring low = LowerRel(rel);
        if (low == wgstate::kSettingsFile || low == wgstate::kModelChoiceFile) {
            string bytes;  // (read through the handle that holds the file until it is deleted)
            return f.ReadBytes(bytes) && wgstate::StampedByWrongGPU(bytes, ToUtf8(low), installId);
        }
        const auto it = names->find(low);
        return it != names->end() && (it->second.empty() || IEq(f.Sha256(), it->second));
    };
}
std::vector<wstring> RemoveStateFolder(const wstring& dir, const Record* m, const wstring& keepFile) {
    return RemoveOwned(StateDir(dir), StateOwned(dir, m), keepFile);
}
// "<prefix><where>: a, b, c and N more" (the first five).
wstring LeftText(const wstring& where, const std::vector<wstring>& rels) {
    wstring names;
    for (size_t i = 0; i < rels.size() && i < 5; ++i) names += (names.empty() ? L"" : L", ") + rels[i];
    if (rels.size() > 5) names += L" and " + std::to_wstring(rels.size() - 5) + L" more";
    return kLeftUnrecorded + where + L": " + names;
}

static std::atomic<CleanupMoveFn> g_cleanupMove{nullptr};  // test seam (record.h SetCleanupMoveForTest)
void SetCleanupMoveForTest(CleanupMoveFn fn) { g_cleanupMove = fn; }

// Returns the line "kept ignore-files.txt ..." when the shared folder's ignore list was put back (convert_ignore.h), else "".
wstring StartCleanup(Context& ctx, const wstring& state, const wstring& self, const wstring& appData) {
    // Detach this generation now. Delayed cleanup must never recursively delete the shared pathname,
    // which a following install can already have recreated with its new model and runtime selection.
    wstring retired, kept;
    if (!appData.empty() && IsDir(appData)) {
        // The name the folder is moved to is free and unique. It was <pid>-<GetTickCount64()> alone; that clock moves in
        // ~16 ms steps, so a second cleanup of the same folder within one step (an install's restore, then the last game's
        // uninstall) aimed at the previous cleanup's folder, which its cmd.exe deletes only seconds later: the move failed
        // with ERROR_ALREADY_EXISTS and the folder was left in place.
        static std::atomic<unsigned> sequence{0};
        for (int k = 0; k < 64; ++k) {
            retired = appData + L".cleanup-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64()) + L"-" +
                      std::to_wstring(sequence++);
            if (GetFileAttributesW(retired.c_str()) == INVALID_FILE_ATTRIBUTES) break;
        }
        // Nothing of WrongGPU's holds a file in the folder here (every write closes its handle, the log opens and closes
        // per line). Another program can, briefly (a scanner reading a file just written): a sharing violation (32) or an
        // access denied (5, how a folder rename fails while a file in it is open elsewhere) is tried again, at most
        // kCleanupMoveTries times, 50-400 ms apart, each wait logged with the error. Any other error is final.
        bool moved = false;
        DWORD err = 0;
        const CleanupMoveFn seam = g_cleanupMove.load();
        const CleanupMoveFn move = seam ? seam : &MoveFileExW;
        for (int t = 0; t < kCleanupMoveTries; ++t) {
            if (move(appData.c_str(), retired.c_str(), MOVEFILE_WRITE_THROUGH)) { moved = true; break; }
            err = GetLastError();
            if ((err != ERROR_SHARING_VIOLATION && err != ERROR_ACCESS_DENIED) || t == kCleanupMoveTries - 1) break;
            const DWORD wait = kCleanupMoveFirstWaitMs << t;
            LogLine(ctx, L"Shared data cleanup: " + appData + L" is in use by another program (error " + std::to_wstring(err) +
                             (err == ERROR_SHARING_VIOLATION ? L", sharing violation" : L", access denied") + L"); trying again in " +
                             std::to_wstring(wait) + L" ms (try " + std::to_wstring(t + 2) + L" of " + std::to_wstring(kCleanupMoveTries) + L")");
            Sleep(wait);
        }
        if (!moved) {
            wstring why = ErrText(err);
            while (!why.empty() && (why.back() == L'\n' || why.back() == L'\r' || why.back() == L' ' || why.back() == L'.')) why.pop_back();
            LogLine(ctx, L"Shared data cleanup deferred: cannot detach " + appData + L" (error " + std::to_wstring(err) + L": " + why +
                             L"); it was left in place.");
            retired.clear();
        } else {
            if (StartsWithI(ctx.logFile, appData + L"\\")) ctx.logFile = retired + ctx.logFile.substr(appData.size());
            // the ignore list stays (an integrator's installer registered its files there; convert_ignore.h): put back, kept
            // beside another program's new one, or left in the moved folder; never deleted
            const ListKept k = PutIgnoreListBack(ctx, retired, appData);
            kept = k.said;
            if (!k.leftIn.empty()) {
                // it could not leave the moved folder: everything else in it goes now; the folder and the list stay, and the
                // hidden cleanup below never deletes that folder's tree
                RemoveTree(retired, k.leftIn);
                LogLine(ctx, L"Shared data cleanup: " + retired + L" stays with the ignore list in it (what was held open in it stays too)");
                retired.clear();
            }
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
    return kept;
}

// ---- the file-ownership rule (record.h) -------------------------------------------------------------------------------
const wchar_t* const kPartlyRestored = L"Partly restored: ";
const wchar_t* const kLeftUnrecorded = L"Left in place (not recorded by the install): ";
bool AnyPartly(const std::vector<wstring>& warnings) {
    for (auto& w : warnings) if (w.rfind(kPartlyRestored, 0) == 0) return true;
    return false;
}

static std::function<void(const char*)> g_restoreStepHook;
void SetRestoreStepHook(std::function<void(const char*)> hook) { g_restoreStepHook = std::move(hook); }
static std::function<bool(const char*)> g_faultHook;
void SetFaultHook(std::function<bool(const char*)> hook) { g_faultHook = std::move(hook); }
bool TestFault(const char* name) { return g_faultHook && g_faultHook(name); }
void RestoreStep(const char* name) {
    if (g_restoreStepHook) g_restoreStepHook(name);
#ifdef WRONGGPU_TESTING
    if (EnvVar(L"WRONGGPU_TEST_STOP_AT") == FromUtf8(name)) ExitProcess(99);  // tests: the process stops here, as if killed
    // tests: the operation fails here (as a failed write would), so its rollback runs
    if (EnvVar(L"WRONGGPU_TEST_FAIL_AT") == FromUtf8(name)) Fail(L"Test failure at " + FromUtf8(name) + L" (WRONGGPU_TEST_FAIL_AT)");
    // tests: a file appears here (another program or the player creating it at this moment), where nothing is
    if (EnvVar(L"WRONGGPU_TEST_CREATE_AT") == FromUtf8(name) && !EnvVar(L"WRONGGPU_TEST_CREATE_FILE").empty()) {
        const wstring f = EnvVar(L"WRONGGPU_TEST_CREATE_FILE");
        EnsureDir(Parent(f));
        const HANDLE h = CreateFileW(f.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            static const char kText[] = "; made by the player during the install (test)\r\n";
            DWORD w = 0;
            WriteFile(h, kText, sizeof kText - 1, &w, nullptr);
            CloseHandle(h);
        }
    }
#endif
}

// (A recorded file that the install created and whose bytes changed since is not WrongGPU's by its name either: WrongGPU's
// programs rewrite wronggpu-rr.ini and the RR plugin's WrongGPU\wronggpu.ini themselves, and so can a player. It stays,
// and the result says so: no operation deletes a file it cannot prove WrongGPU's by its bytes.)
static const FfxSlot* SlotOf(const Entry& e) {
    for (auto& sl : kSlots) if (IEq(e.rel, e.action == L"rename" ? sl.original : sl.name)) return &sl;
    return nullptr;
}

Holds WhatIsThere(Context& ctx, const wstring& dir, const Entry& e, const wstring& h) {
    if (h.empty()) return Holds::Missing;
    if (e.action == L"disable") return IEq(h, e.orig) ? Holds::Original : Holds::Other;
    if (e.action == L"rename") return IEq(h, e.inst) ? Holds::Original : Holds::Other;  // the game's own file, renamed aside
    if (e.action != L"copy") return Holds::Other;
    if (!e.inst.empty() && IEq(h, e.inst)) return Holds::Ours;
    if (e.existed && !e.orig.empty() && IEq(h, e.orig)) return Holds::Original;
    const wstring t = Target(dir, e.rel);
    wstring info;
    if (const FfxSlot* sl = SlotOf(e)) return ClassifySlot(*sl, t, nullptr, info) == FFX_OURS ? Holds::Ours : Holds::Other;  // the slot exception
    if (!EndsWithI(e.rel, L".ini") && KnownReleaseFile(ctx, t)) return Holds::Ours;  // a WrongGPU release's file (size and SHA-256)
    return Holds::Other;
}

wstring KeptOriginal(const wstring& dir, const wstring& rel, const wstring& source, const wstring& sha) {
    const wstring base = Join(dir, kKeptDirName);
    for (auto& gen : ListDirs(base)) {  // kept already (a second run after a stop): that copy is the one
        const wstring p = Join(gen, rel);
        if (IsFile(p) && IEq(Sha256File(p), sha)) return p;
    }
    SYSTEMTIME st; GetLocalTime(&st);
    wchar_t stamp[32];
    swprintf(stamp, 32, L"%04d%02d%02d-%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    // never over another generation: the copy is created only where nothing is (CopyNew), the next free name otherwise
    for (int n = 1; n < 1000; ++n) {
        const wstring gen = Join(base, n == 1 ? wstring(stamp) : wstring(stamp) + L"-" + std::to_wstring(n));
        const wstring p = Join(gen, rel);
        if (Exists(p)) continue;
        EnsureDir(Parent(p));
        DWORD err = 0;
        if (!CopyNew(source, p, &err)) {
            if (err == ERROR_FILE_EXISTS || err == ERROR_ALREADY_EXISTS) continue;  // another file appeared there a moment ago
            Fail(L"Could not keep the original of " + rel + L" in " + Parent(p) + L": " + ErrText(err) + L" Nothing more was changed.");
        }
        if (!IEq(Sha256File(p), sha)) Fail(L"Could not keep the original of " + rel + L" in " + Parent(p) + L". Nothing more was changed.");
        return p;
    }
    Fail(L"Could not keep the original of " + rel + L" in " + base + L": no free name. Nothing more was changed.");
}

// The file can be deleted or replaced now (no other program holds it without sharing deletion; read-only is fine).
static bool Replaceable(const wstring& path) {
    HANDLE h = CreateFileW(path.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return GetLastError() == ERROR_FILE_NOT_FOUND || GetLastError() == ERROR_PATH_NOT_FOUND;
    CloseHandle(h);
    return true;
}

// A WrongGPU binary: what a game loads (DLLs and the OptiScaler plugins).
bool IsBinaryName(const wstring& rel) { return EndsWithI(rel, L".dll") || EndsWithI(rel, L".asi"); }

// The binaries of an install the operation guard holds: every copy entry of WrongGPU's (owner wg / wg-optibundle) that is a
// binary, at its path in the game folder (whatever is there now: a file the guard cannot open refuses the operation).
static std::vector<wstring> GuardedBinaries(const wstring& dir, const Record& m) {
    std::vector<wstring> r;
    for (auto& e : m.e)
        if (e.action == L"copy" && (e.owner == L"wg" || e.owner == L"wg-optibundle" || e.owner.empty()) && IsBinaryName(e.rel)) {
            const wstring p = Target(dir, e.rel);
            bool seen = false;  // each file once: a second open of a held file would refuse the operation against its own hold
            for (auto& q : r) seen = seen || IEq(FullPath(q), FullPath(p));
            if (!seen) r.push_back(p);
        }
    return r;
}

static thread_local OperationGate* t_gate = nullptr;  // the innermost gate of this thread
OperationGate::OperationGate(const wstring& d, const std::vector<wstring>& binaries) : dir(d), guard(d, binaries), prev(t_gate) { t_gate = this; }
OperationGate::~OperationGate() { t_gate = prev; }

#ifdef WRONGGPU_TESTING
// Test builds: which processes have a file open (Restart Manager), for the log when the gate refuses.
static wstring HoldersOf(const wstring& path) {
    DWORD session = 0;
    WCHAR key[CCH_RM_SESSION_KEY + 1] = {};
    if (RmStartSession(&session, 0, key) != ERROR_SUCCESS) return L"?";
    LPCWSTR files[] = {path.c_str()};
    wstring out;
    if (RmRegisterResources(session, 1, files, 0, nullptr, 0, nullptr) == ERROR_SUCCESS) {
        UINT needed = 0, n = 16;
        RM_PROCESS_INFO info[16];
        DWORD reasons = 0;
        if (RmGetList(session, &needed, &n, info, &reasons) == ERROR_SUCCESS)
            for (UINT i = 0; i < n; ++i)
                out += L" pid " + std::to_wstring(info[i].Process.dwProcessId) + (info[i].Process.dwProcessId == GetCurrentProcessId() ? L" (this process)" : L"") +
                       L" " + info[i].strAppName + L";";
    }
    RmEndSession(session);
    return out.empty() ? L" none listed" : out;
}
#endif

std::unique_ptr<OperationGate> GuardOperation(const wstring& dir, const Record* m, bool holdBinaries) {
    if (t_gate && IEq(FullPath(t_gate->dir), FullPath(dir))) return nullptr;  // this thread's operation holds the gate already
    Record loaded;
    if (!m && LoadRecord(RecordPath(dir), loaded) && !loaded.legacy) m = &loaded;
    auto gate = std::make_unique<OperationGate>(dir, m && holdBinaries ? GuardedBinaries(dir, *m) : std::vector<wstring>{});
    if (!gate->guard.ok()) {
        wstring which = gate->guard.which().empty() ? wstring() : L" (" + BaseName(gate->guard.which()) + L")";
#ifdef WRONGGPU_TESTING
        if (!gate->guard.which().empty()) which += L" [test build, holders:" + HoldersOf(gate->guard.which()) + L"]";
#endif
        Fail(L"Nothing was changed: " + FromUtf8(gate->guard.words()) + which + L".", EXIT_REFUSED);
    }
    return gate;
}

bool GateHolds(const wstring& path) { return t_gate && t_gate->guard.Holds(path); }

// A binary written (or replaced) by the operation: through the gate (PutHeld: renamed in complete, then held) when this thread
// holds one, else plainly. Throws on failure; a load in the gap (sharing violation) stops the operation.
static void FailPut(const wstring& path, DWORD e) {
    if (e == ERROR_SHARING_VIOLATION)
        Fail(L"Stopped: " + FromUtf8(wgstate::GuardWords(wgstate::GuardResult::GameRunning)) + L" (" + BaseName(path) +
             L"). What was done so far stays recorded; run it again once the game is closed.", EXIT_REFUSED);
    Fail(L"Cannot write " + path + L": " + ErrText(e));
}
// A file at the path that the gate does not hold (one the install decided to replace that the record did not list, such as
// an earlier copy in REFramework's mirror): opened as the gate opens its binaries (a program that has it loaded, or open
// without sharing writes, stops the operation) and deleted through that handle, so the new file can be renamed in and held.
static DWORD FreeUnheld(const wstring& path) {
    if (GateHolds(path) || !Exists(path)) return ERROR_SUCCESS;
    const HANDLE h = CreateFileW(wgstate::ExtendedPath(path).c_str(), DELETE | GENERIC_READ | FILE_WRITE_DATA, FILE_SHARE_READ, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return GetLastError();
    FILE_DISPOSITION_INFO_EX d{};
    d.Flags = FILE_DISPOSITION_FLAG_DELETE | FILE_DISPOSITION_FLAG_POSIX_SEMANTICS;
    const DWORD e = SetFileInformationByHandle(h, FileDispositionInfoEx, &d, sizeof d) ? ERROR_SUCCESS : GetLastError();
    CloseHandle(h);
    return e;
}
void PutBinary(const wstring& path, const void* data, size_t size) {
    if (!t_gate) { WriteFileAtomic(path, data, size); return; }
    EnsureDir(Parent(path));
    if (const DWORD fe = FreeUnheld(path)) FailPut(path, fe);
    std::wstring oldLeft;
    const DWORD e = t_gate->guard.PutHeld(path, data, size, &oldLeft);
    if (e == ERROR_SUCCESS && !oldLeft.empty()) t_gate->oldLeft.push_back(oldLeft);
    if (e == ERROR_SUCCESS) return;
    FailPut(path, e);
}
void PutBinaryFrom(const wstring& path, const wstring& source) {
    if (!t_gate) { CopyChecked(source, path); return; }
    EnsureDir(Parent(path));
    if (const DWORD fe = FreeUnheld(path)) FailPut(path, fe);
    std::wstring oldLeft;
    const DWORD e = t_gate->guard.PutHeldFrom(path, source, &oldLeft);
    if (e == ERROR_SUCCESS && !oldLeft.empty()) t_gate->oldLeft.push_back(oldLeft);
    if (e == ERROR_SUCCESS) return;
    FailPut(path, e);
}
DWORD DeleteHeldBinary(const wstring& path) { return t_gate ? t_gate->guard.DeleteHeld(path) : ERROR_NOT_FOUND; }
std::vector<wstring> DoRestore(Context& ctx, const wstring& dir, Ui& ui, bool failedInstall, bool keepState, std::map<wstring, wstring>* keptWarnings) {
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy) Fail(L"No WrongGPU installation record was found in " + dir + L".", EXIT_NOT_INSTALLED);
    if (!IEq(m.game, dir)) Fail(L"The installation record belongs to a different folder: " + m.game);
    // #219: never while a running program has WrongGPU loaded (wg_state.h, the loaded protocol: the state lock is held for the
    // whole restore, so no game registers in between; a failed install's own restore runs under its install's guard)
    const auto gate = failedInstall ? nullptr : GuardOperation(dir, &m);
    std::vector<wstring> warnings;
    int total = (int)m.e.size() + 2, done = 0;

    // What to do with each entry, decided before anything changes.
    enum class Do { Nothing, Remove, PutBack, Leave };
    struct Plan { Do what = Do::Nothing; wstring now, source; bool keep = false; };  // now: the hash the decision was made on
    std::vector<Plan> plan(m.e.size());
    auto leaveWarning = [&](const Entry& e, const wstring& t, const wstring& keptAt, bool linked = false) {
        // (not WrongGPU's settings files: the headers WrongGPU writes in WrongGPU\wronggpu.ini and wronggpu-asi.ini name OptiScaler)
        const bool settingsFile = IsRrSettingsFile(e.rel) || IEq(BaseName(e.rel), L"wronggpu-asi.ini");
        const Fingerprint fp = e.action == L"copy" && !IsSlotName(e.rel) && !settingsFile ? IdentifyFile(t) : Fingerprint();
        wstring w;
        if (linked) w = L"Left in place (the file has another name, a hard link; writing through it would change that name too): " + e.rel;
        else if (e.action == L"copy" && IsSlotName(e.rel)) w = L"The game replaced " + e.rel + L" (probably an update); its new file was kept.";
        else if (e.action == L"disable") w = e.rel + L" exists again with different content; it was left as it is.";
        else if (fp.tool == Tool::OptiScaler) w = L"Left in place: " + e.rel + L" is now another setup's OptiScaler (not the file WrongGPU installed).";
        else w = L"Left in place (changed since install): " + e.rel;
        if (!keptAt.empty()) w = kPartlyRestored + w + L" The " + e.rel + L" from before WrongGPU is kept in " + keptAt;
        return w;
    };

    if (!failedInstall) {
        ui.step(done, total, L"Checking the backup");
        for (size_t i = 0; i < m.e.size(); ++i) {
            const Entry& e = m.e[i];
            Plan& p = plan[i];
            if (e.action == L"dir") continue;
            const wstring t = Target(dir, e.rel), b = EntryBackupPath(dir, e);
            // the player's original: the backup, or (a game FSR DLL) its renamed copy beside WrongGPU's
            const bool hasOriginal = e.existed || e.action == L"disable";
            if (hasOriginal) {
                p.source = b;
                if (!Exists(b) || !IEq(Sha256File(b), e.orig)) {
                    const FfxSlot* sl = e.action == L"copy" ? SlotOf(e) : nullptr;
                    const wstring kept = sl ? Target(dir, sl->original) : wstring();
                    if (kept.empty() || !IEq(Sha256File(kept), e.orig)) Fail(L"The backup of " + e.rel + L" is missing or damaged. Nothing was changed.");
                    p.source = kept;
                }
            }
            if (Exists(t)) {
                p.now = Sha256File(t);
                // a file that cannot be read (another program holds it open without sharing, such as the game) is never taken
                // for "changed since the install": the restore stops before anything changes
                if (p.now.empty()) Fail(L"Cannot read " + e.rel + L": another program may have it open (the game?). Close it and try again. Nothing was changed.", EXIT_REFUSED);
            }
            const Holds holds = WhatIsThere(ctx, dir, e, p.now);
            if (e.action == L"rename") {  // the renamed game file: handled with its slot below
                if (Exists(t) && !Replaceable(t)) Fail(L"Cannot change " + e.rel + L": another program has it open (the game?). Close it and try again. Nothing was changed.", EXIT_REFUSED);
                continue;
            }
            if (e.action == L"copy" && e.owner == L"adopted") continue;  // the player's file, used but never deleted
            switch (holds) {
                case Holds::Ours: p.what = hasOriginal ? Do::PutBack : Do::Remove; break;
                case Holds::Missing: p.what = hasOriginal ? Do::PutBack : Do::Nothing; break;
                case Holds::Original: p.what = Do::Nothing; break;
                case Holds::Other: p.what = Do::Leave; p.keep = hasOriginal; break;
            }
            if ((p.what == Do::Remove || p.what == Do::PutBack) && Exists(t) && !GateHolds(t) && !Replaceable(t))  // (a held binary: the gate changes it)
                Fail(L"Cannot change " + e.rel + L": another program has it open (the game?). Close it and try again. Nothing was changed.", EXIT_REFUSED);
        }
    } else {
        // a half-finished install: what it wrote goes, what was there comes back. By the same rule as any restore: only a file
        // whose bytes are the installed ones (or the original's, which stays) is touched; anything else at the path stays and
        // is said (a file another program put there while the install ran; a copy cut off halfway).
        for (size_t i = 0; i < m.e.size(); ++i) {
            const Entry& e = m.e[i];
            Plan& p = plan[i];
            if (e.action == L"dir" || e.action == L"rename") continue;
            p.source = EntryBackupPath(dir, e);
            if (e.action == L"copy" && e.owner == L"adopted") continue;
            const bool hasOriginal = e.action == L"disable" || e.existed;
            const wstring t = Target(dir, e.rel);
            if (Exists(t)) p.now = Sha256File(t);
            if (Exists(t) && p.now.empty()) { p.what = Do::Leave; p.keep = hasOriginal && Exists(p.source); continue; }  // unreadable: not touched
            switch (WhatIsThere(ctx, dir, e, p.now)) {
                case Holds::Ours: p.what = hasOriginal ? Do::PutBack : Do::Remove; break;
                case Holds::Missing: p.what = hasOriginal ? Do::PutBack : Do::Nothing; break;
                case Holds::Original: p.what = Do::Nothing; break;
                case Holds::Other: p.what = Do::Leave; p.keep = hasOriginal && Exists(p.source); break;
            }
        }
    }

    // The originals that cannot go back are kept first, where they survive (before any file changes).
    for (size_t i = 0; i < m.e.size(); ++i) {
        const Entry& e = m.e[i];
        Plan& p = plan[i];
        if (p.what != Do::Leave) continue;
        const wstring keptAt = p.keep ? KeptOriginal(dir, e.rel, p.source, e.orig) : wstring();
        warnings.push_back(leaveWarning(e, Target(dir, e.rel), keptAt));
        if (keptWarnings) (*keptWarnings)[Lower(e.rel)] = warnings.back();
        LogLine(ctx, warnings.back());
    }
    RestoreStep("kept");

    for (size_t k = m.e.size(); k-- > 0;) {  // reverse order of installation
        const Entry& e = m.e[k];
        Plan& p = plan[k];
        ui.step(++done, total, L"Restoring " + e.rel);
        const wstring t = Target(dir, e.rel);
        if (e.action == L"dir") {
            if (IsDir(t) && RemoveDirectoryW(t.c_str())) ui.log(L"removed folder " + e.rel);
            continue;
        }
        if (e.action == L"rename") {
            // the game's FSR DLL, renamed aside: back under its name when WrongGPU's DLL is still under it (no entry or no
            // backup for that one: the renamed file is the game's only copy)
            const FfxSlot* sl = SlotOf(e);
            const wstring slot = sl ? Join(dir, sl->name) : wstring();
            wstring info;
            if (sl && Exists(t) && (e.inst.empty() || IEq(Sha256File(t), e.inst))) {
                // WrongGPU's DLL is held (pinned) while it is recognised and deleted through that handle; the game's file then
                // takes the name without replacing anything (a file that appears at the name in between stays, and the branch
                // below keeps the game's file). A name that is empty (a stop between the two steps, or the DLL removed) gets it.
                PinnedFile held;
                DWORD err = 0;
                if (GateHolds(slot)) {  // #219: WrongGPU's DLL in the slot is held by the operation's gate: deleted through it
                    if (ClassifySlot(*sl, slot, nullptr, info) == FFX_OURS) {
                        const DWORD de = DeleteHeldBinary(slot);
                        if (de != ERROR_SUCCESS) Fail(L"Cannot put " + wstring(sl->name) + L" back: " + ErrText(de));
                        if (MoveFileExW(t.c_str(), slot.c_str(), MOVEFILE_WRITE_THROUGH)) ui.log(L"restored " + wstring(sl->name) + L" from " + e.rel);
                        else if (GetLastError() != ERROR_ALREADY_EXISTS && GetLastError() != ERROR_FILE_EXISTS)
                            Fail(L"Cannot put " + wstring(sl->name) + L" back: " + ErrText(GetLastError()));
                    }
                } else if (held.Open(slot, false, &err)) {
                    if (ClassifySlot(*sl, slot, nullptr, info) == FFX_OURS) {
                        RestoreStep("locked");
                        if (!held.Delete(&err)) Fail(L"Cannot put " + wstring(sl->name) + L" back: " + ErrText(err));
                        if (MoveFileExW(t.c_str(), slot.c_str(), MOVEFILE_WRITE_THROUGH)) ui.log(L"restored " + wstring(sl->name) + L" from " + e.rel);
                        else if (GetLastError() != ERROR_ALREADY_EXISTS && GetLastError() != ERROR_FILE_EXISTS)
                            Fail(L"Cannot put " + wstring(sl->name) + L" back: " + ErrText(GetLastError()));
                    }
                } else if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
                    if (MoveFileExW(t.c_str(), slot.c_str(), MOVEFILE_WRITE_THROUGH)) ui.log(L"restored " + wstring(sl->name) + L" from " + e.rel);
                } else if (err != ERROR_NOT_SUPPORTED) {
                    Fail(L"Cannot put " + wstring(sl->name) + L" back: another program has it open.");
                }
            }
            // still there: the game's file under WrongGPU's name for it. It goes only when its bytes are safe elsewhere: back
            // under the game's name, or kept (a game update put another file under that name)
            if (sl && Exists(t) && !e.inst.empty() && IEq(Sha256File(t), e.inst)) {
                const wstring at = slot;
                if (!(IsFile(at) && IEq(Sha256File(at), e.inst))) {
                    const wstring keptAt = KeptOriginal(dir, sl->name, t, e.inst);
                    warnings.push_back(kPartlyRestored + wstring(L"The game replaced ") + sl->name + L" (probably an update); its new file was kept. The " +
                                       sl->name + L" from before WrongGPU is kept in " + keptAt);
                }
                DWORD derr = 0;
                const DeleteIf r = DeleteIfSha(t, {e.inst}, &derr);  // (hashed and deleted through one handle)
                if (r == DeleteIf::Failed) Fail(L"Cannot delete " + t + L": " + ErrText(derr));
                if (r == DeleteIf::Changed) {
                    warnings.push_back(L"Left in place (changed since install): " + e.rel);
                    LogLine(ctx, warnings.back());
                } else ui.log(L"removed " + e.rel + L" (the game's file is back under its name, or kept)");
            }
            continue;
        }
        if (p.what == Do::Nothing || p.what == Do::Leave) { if (p.what == Do::Leave) ui.log(L"kept " + e.rel + L" as it is"); continue; }
        // The file is checked again, and changed, through one handle (PinnedFile, util.h): opened once without sharing for
        // writing, renaming or deleting, hashed through that handle, and deleted or overwritten through it. A file that is not
        // what the plan saw (it changed after the check) is left as it is, its original kept, and said; one that appeared at
        // a path the plan found empty is never overwritten (CopyNew creates only where nothing is).
        auto leaveChanged = [&](bool linked = false) {
            const wstring keptAt = p.what == Do::PutBack && (!failedInstall || Exists(p.source)) ? KeptOriginal(dir, e.rel, p.source, e.orig) : wstring();
            warnings.push_back(leaveWarning(e, t, keptAt, linked));
            if (keptWarnings) (*keptWarnings)[Lower(e.rel)] = warnings.back();
            LogLine(ctx, warnings.back());
        };
        auto isGone = [](DWORD err) { return err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND; };
        auto cannot = [&](DWORD err) {
            if (err == ERROR_SHARING_VIOLATION || err == ERROR_ACCESS_DENIED || err == ERROR_LOCK_VIOLATION)
                Fail(L"Cannot change " + e.rel + L": another program has it open (the game?). Close it and try again.", EXIT_REFUSED);
            Fail(L"Cannot change " + e.rel + L": " + ErrText(err));
        };
        if (GateHolds(t)) {  // #219: a binary the operation's gate holds (nothing could change it since the plan hashed it)
            RestoreStep("locked");  // (held and checked: the tests try to change it here, and cannot)
            if (p.what == Do::Remove) {
                const DWORD de = DeleteHeldBinary(t);
                if (de != ERROR_SUCCESS) Fail(L"Cannot delete " + t + L": " + ErrText(de));
                ui.log(L"removed " + e.rel);
            } else {  // PutBack: WrongGPU's file deleted through its handle, then the original copied in, only if the name is free;
                      // the original is the game's or the player's file, so the gate does not hold it (a later step of the same
                      // operation, such as the install of a repair, may rename it); a failed copy leaves the backup for another run
                if (failedInstall && !Exists(p.source)) continue;
                const DWORD de = DeleteHeldBinary(t);
                if (de != ERROR_SUCCESS) Fail(L"Cannot delete " + t + L": " + ErrText(de));
                if (!CopyFileW(p.source.c_str(), t.c_str(), TRUE)) {
                    const DWORD ce = GetLastError();
                    Fail(L"Cannot put back " + t + L": " + ErrText(ce) + L" Its original is kept in " + p.source + L"; run the uninstall again.");
                }
                if (!failedInstall && !IEq(Sha256File(t), e.orig)) Fail(L"Restore verification failed for " + e.rel);
                ui.log(L"restored " + e.rel);
            }
            continue;
        }
        PinnedFile held;
        DWORD err = 0;
        const bool opened = held.Open(t, p.what == Do::PutBack, &err);
        if (!opened && err == ERROR_NOT_SUPPORTED) { leaveChanged(); continue; }  // a folder or a link there now: not WrongGPU's file
        if (!opened && err == ERROR_TOO_MANY_LINKS) { leaveChanged(true); continue; }  // a hard link: never written through, the original is kept
        if (!opened && !isGone(err)) cannot(err);
        if (opened) {
            const wstring h = held.Sha256();
            if (h.empty()) Fail(L"Cannot read " + e.rel + L": another program may have it open (the game?). Close it and try again.", EXIT_REFUSED);
            if (!IEq(h, p.now)) { held.Close(); leaveChanged(); continue; }
            RestoreStep("locked");  // the file is held and checked: the tests try to change it here (they cannot)
            if (p.what == Do::Remove) {
                if (!held.Delete(&err)) Fail(L"Cannot delete " + t + L": " + ErrText(err));
                ui.log(L"removed " + e.rel);
            } else {  // PutBack
                if (failedInstall && !Exists(p.source)) continue;
                if (!held.Overwrite(p.source, e.orig, &err)) Fail(L"Cannot restore " + e.rel + L": " + ErrText(err));
                if (!failedInstall && !IEq(held.Sha256(), e.orig)) Fail(L"Restore verification failed for " + e.rel);
                ui.log(L"restored " + e.rel);
            }
            continue;
        }
        // gone since the plan: nothing to lose; an original simply goes back, created where nothing is
        if (p.what == Do::Remove || (failedInstall && !Exists(p.source))) continue;
        RestoreStep("absent");  // nothing is there: the tests create a file here (CopyNew must not overwrite it)
        EnsureDir(Parent(t));
        if (!CopyNew(p.source, t, &err)) {
            if (err == ERROR_FILE_EXISTS || err == ERROR_ALREADY_EXISTS) { leaveChanged(); continue; }  // a file appeared there a moment ago
            Fail(L"Cannot copy " + p.source + L" to " + t + L": " + ErrText(err));
        }
        if (!failedInstall && !IEq(Sha256File(t), e.orig)) Fail(L"Restore verification failed for " + e.rel);
        ui.log(L"restored " + e.rel);
    }
    RestoreStep("restored");
    // REFramework's copies of the install's root DLLs (REFramework came, or made its mirror, after the install; an install
    // from before this build): the recorded ones were restored above like any recorded file. A copy that is NOT in the record
    // is never deleted, not even when it is byte for byte a file the install wrote (REFramework or the player made that copy,
    // and nothing says they agree to its removal): it stays and is said, because the game can go on loading WrongGPU's file
    // from there (gamerules.h).
    std::vector<wstring> mirrorLeft;
    if (IEq(FullPath(dir), FullPath(m.exe.empty() ? dir : Parent(m.exe))) && IsDir(Join(dir, kMirrorDir))) {
        std::set<wstring> tracked;
        for (auto& e : m.e) if (IsMirrorRel(e.rel)) tracked.insert(Lower(e.rel));
        for (size_t k = 0; k < m.e.size(); ++k) {
            const Entry& e = m.e[k];
            if (e.action != L"copy" || !MirroredRel(e.rel) || e.inst.empty() || plan[k].what == Do::Leave || tracked.count(Lower(MirrorRel(e.rel)))) continue;
            const wstring t = Target(dir, MirrorRel(e.rel));
            if (IsFile(t)) mirrorLeft.push_back(MirrorRel(e.rel));
        }
        for (auto& f : mirrorLeft) LogLine(ctx, L"kept " + f + L" (REFramework's copy of a WrongGPU file; the install did not record it, so it is not deleted)");
    }
    ui.step(++done, total, L"Cleaning up");
    if (!failedInstall) {
        // wronggpu-* files in the folders the record lists that the install did not record: never deleted (nothing proves them
        // WrongGPU's: a player's notes or another program's file can carry the name); each is logged, and an uninstall says so
        const std::vector<wstring> left = UnrecordedRuntimeFiles(dir, m);
        for (auto& f : left) LogLine(ctx, L"kept " + f + L" (named like WrongGPU's own files, but the install did not record it, so it is not deleted)");
        if (!left.empty() && !keepState) {
            wstring names;
            for (size_t i = 0; i < left.size() && i < 5; ++i) {
                const wstring& f = left[i];
                names += (names.empty() ? L"" : L", ") + (StartsWithI(f, dir + L"\\") ? f.substr(dir.size() + 1) : f);
            }
            if (left.size() > 5) names += L" and " + std::to_wstring(left.size() - 5) + L" more";
            warnings.push_back(kLeftUnrecorded + names);
            ui.log(warnings.back());
        }
        if (!mirrorLeft.empty() && !keepState) { warnings.push_back(LeftText(L"REFramework's copies", mirrorLeft)); ui.log(warnings.back()); }
        for (size_t k = m.e.size(); k-- > 0;)  // a folder the install made that held only those files
            if (m.e[k].action == L"dir" && IsDir(Target(dir, m.e[k].rel)) && RemoveDirectoryW(Target(dir, m.e[k].rel).c_str()))
                ui.log(L"removed folder " + m.e[k].rel);
    }
    // an OptiScaler.ini that stays after the uninstall (another setup's, or one changed since) and names WrongGPU's files:
    // that setup uses them and may stop working now that WrongGPU's copies are gone
    if (!failedInstall && !keepState) {
        const wstring exeDir = m.exe.empty() ? dir : Parent(m.exe);
        std::set<wstring> seen;
        for (const wstring& d : {dir, exeDir, Join(exeDir, L"OptiScaler")}) {
            const wstring p = Join(d, L"OptiScaler.ini");
            string text;
            if (!seen.insert(Lower(FullPath(p))).second || !ReadAll(p, text)) continue;
            for (auto& c : text) c = (char)tolower((unsigned char)c);
            if (text.find("wronggpu") == string::npos) continue;
            const wstring shown = p.size() > dir.size() + 1 && StartsWithI(p, dir + L"\\") ? p.substr(dir.size() + 1) : p;
            warnings.push_back(L"Another setup's " + shown + L" still names WrongGPU's files: that setup may need them, and WrongGPU's copies were removed with it.");
        }
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

    // The record goes before the backup: a stop after this point leaves every original at its path or kept (above), and a
    // backup folder without a record (nothing refers to it); a stop before it leaves the record and the backup for a
    // second run, which finds the files restored so far as they should be.
    const wstring state = StateDir(dir);
    if (!DeleteFileW(RecordPath(dir).c_str()) && IsFile(RecordPath(dir))) Fail(L"Cannot remove " + RecordPath(dir) + L": " + ErrText(GetLastError()));
    RestoreStep("record-removed");
    // the shared folder goes, its ignore-files.txt stays (StartCleanup): said in the result
    auto cleanup = [&](const wstring& st, const wstring& sf) {
        const wstring kept = StartCleanup(ctx, st, sf, appData);
        if (!kept.empty()) { ui.log(kept); warnings.push_back(kept); }
    };
    // The state folder: what WrongGPU wrote goes (RemoveStateFolder); anything else in it, and the folder, stay and are said.
    const auto stateOwned = StateOwned(dir, &m);
    if (keepState) {
        for (auto& f : RemoveOwned(Join(state, L"backup"), [&](const wstring& rel, PinnedFile& h) { return stateOwned(L"backup\\" + rel, h); }))
            LogLine(ctx, L"kept " + Join(Join(state, L"backup"), f) + L" (not the backup of a recorded file, or changed since)");
        return warnings;
    }
    const wstring self = SelfPath();
    std::vector<wstring> stay;
    if (IEq(Parent(self), state)) {
        stay = RemoveOwned(state, stateOwned, self);
        for (auto& f : stay) LogLine(ctx, L"kept " + Join(state, f) + L" (not WrongGPU's, changed since, or in use)");
        if (!stay.empty()) { warnings.push_back(LeftText(L"wronggpu-setup", stay)); ui.log(warnings.back()); }
        cleanup(state, self);
    } else {
        stay = RemoveOwned(state, stateOwned);
        for (auto& f : stay) LogLine(ctx, L"kept " + Join(state, f) + L" (not WrongGPU's, changed since, or in use)");
        if (!stay.empty()) { warnings.push_back(LeftText(L"wronggpu-setup", stay)); ui.log(warnings.back()); }
        if (!appData.empty()) cleanup(L"", L"");
    }
    return warnings;
}

static bool SettingsName(const wstring& name, bool folder) {
    return folder ? IEq(name, L"previous") || IEq(name, L"rr-config")
                  : IEq(name, L"profile.ini") || IEq(name, wgstate::kSettingsFile) || IEq(name, wgstate::kModelChoiceFile);  // (#219: the state files)
}
bool StateHasSettings(const wstring& dir) {
    const wstring state = StateDir(dir);
    for (auto& f : ListFiles(state, L"*")) if (SettingsName(BaseName(f), false)) return true;
    for (auto& d : ListDirs(state)) if (SettingsName(BaseName(d), true)) return true;
    return false;
}
wstring CopyBackup(Context& ctx, const wstring& dir, Entry& e, const wstring& source, RunJournal* j, std::vector<wstring>* warnings) {
    for (int n = 0; n < 100; ++n) {
        const wstring name = n == 0 ? e.rel : e.rel + L"." + std::to_wstring(n);
        const wstring b = BackupPath(dir, name);
        if (Exists(b)) continue;  // something is there that this install did not make: never overwritten
        const size_t mark = j ? j->created.size() : 0;
        if (j) JournalCreate(*j, b, false);
        EnsureDir(Parent(b));
        DWORD err = 0;
        if (!CopyNew(source, b, &err)) {
            if (j && j->created.size() > mark && j->created.back().first == b) j->created.pop_back();  // (not this run's: not created)
            if (err == ERROR_FILE_EXISTS || err == ERROR_ALREADY_EXISTS) continue;  // it appeared a moment ago: the next name
            Fail(L"Cannot copy " + source + L" to " + b + L": " + ErrText(err));
        }
        if (Sha256File(b) != e.orig) Fail(L"Backup verification failed for " + e.rel);
        e.backup = n == 0 ? wstring() : name;
        if (n) {
            const wstring w = L"Left in place (not this install's; an earlier uninstall kept it): " + Join(Join(kStateDirName, L"backup"), e.rel) +
                              L"; this install's backup of " + e.rel + L" is " + Join(Join(kStateDirName, L"backup"), name);
            LogLine(ctx, w);
            if (warnings) warnings->push_back(w);
        }
        return b;
    }
    Fail(L"No free name for the backup of " + e.rel + L" in " + Join(StateDir(dir), L"backup"));
}
wstring NewInstallId() {
    std::random_device rd;  // (the CRT's system random source)
    wchar_t buf[33] = {};
    for (int i = 0; i < 4; ++i) swprintf(buf + i * 8, 9, L"%08x", static_cast<unsigned>(rd()));
    return buf;
}
void JournalCreate(RunJournal& j, const wstring& path, bool folder) {
    if (Exists(path)) return;  // there before this run: never this run's
    const wstring up = Parent(path);
    if (!up.empty() && up != path && !Exists(up)) JournalCreate(j, up, true);
    j.created.push_back({path, folder});
}
std::vector<wstring> RollbackCreated(Context& ctx, const wstring& state, RunJournal& j) {
    std::vector<wstring> left;
    try {
        for (auto it = j.created.rbegin(); it != j.created.rend(); ++it) {
            if (it->second) RemoveDirectoryW(it->first.c_str());  // (only when empty: anything else in it stays)
            else if (IsFile(it->first)) { SetFileAttributesW(it->first.c_str(), FILE_ATTRIBUTE_NORMAL); DeleteFileW(it->first.c_str()); }
        }
        j.created.clear();
        j.done = true;
        std::vector<wstring> stack{state};
        while (!stack.empty()) {
            const wstring d = stack.back();
            stack.pop_back();
            for (auto& f : ListFiles(d, L"*")) left.push_back(f.substr(state.size() + 1));
            for (auto& s : ListDirs(d)) stack.push_back(s);
        }
        for (auto& f : left) LogLine(ctx, L"kept " + Join(state, f) + L" (not made by this run: the rollback takes back only what it made)");
    } catch (...) {}
    return left;
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
            // checked and changed through one handle (PinnedFile); only the bytes the check above accepted (installed, original)
            DWORD err = 0;
            if (e.existed) {
                PinnedFile held;
                const bool opened = held.Open(t, true, &err);
                if (!opened && err != ERROR_FILE_NOT_FOUND && err != ERROR_PATH_NOT_FOUND) {
                    warnings.push_back(err == ERROR_TOO_MANY_LINKS ? L"Older install: left in place (a hard link: not written through): " + e.rel
                                                                    : L"Older install: left in place (changed since install): " + e.rel);
                    continue;
                }
                if (opened) {
                    const wstring h = held.Sha256();
                    if (h != e.inst && h != e.orig) { warnings.push_back(L"Older install: left in place (changed since install): " + e.rel); continue; }
                    if (!held.Overwrite(b, e.orig, &err)) Fail(L"Cannot restore " + e.rel + L": " + ErrText(err));
                } else {
                    EnsureDir(Parent(t));
                    if (!CopyNew(b, t, &err)) { warnings.push_back(L"Older install: left in place (changed since install): " + e.rel); continue; }
                }
                if (Sha256File(t) != e.orig) Fail(L"Restore verification failed for " + e.rel);
                ui.log(L"older install: restored " + e.rel);
            } else {
                const DeleteIf r = DeleteIfSha(t, {e.inst, e.orig}, &err);
                if (r == DeleteIf::Failed) Fail(L"Cannot delete " + t + L": " + ErrText(err));
                if (r == DeleteIf::Changed) warnings.push_back(L"Older install: left in place (changed since install): " + e.rel);
                else if (r == DeleteIf::Deleted) ui.log(L"older install: removed " + e.rel);
            }
        } else if (!Exists(t)) {
            EnsureDir(Parent(t));
            DWORD err = 0;
            if (CopyNew(b, t, &err)) ui.log(L"older install: restored " + e.rel);
        }
    }
    for (const wchar_t* d : {L"dlss45-model\\gpu-build", L"dlss45-model\\exposure-build", L"dlss45-model"}) RemoveDirectoryW(Join(dir, d).c_str());
    std::set<wstring> recorded;
    for (auto& e : m.e) recorded.insert(Lower(FullPath(Target(dir, e.rel))));
    // the older install's own log and report names: never deleted by name (a file that is not in the record is not provably the
    // older install's); each is logged and listed in the result
    std::vector<wstring> unrecorded;
    for (const wstring& d : {dir, GameRoot(dir)})
        for (auto pat : {L"dlss45-*.log", L"dlss45-*.log.old", L"dlss45-report.txt", L"dlss45-report.json", L"dlss45-dred.txt", L"dlss45-crash-*.txt",
                         L"dlss45-crash-*.dmp", L"dlss45-capture-*.bin", L"dlss45-capture-*.txt", L"dlss45-capture-*.jpg"})
            for (auto& f : ListFiles(d, pat))
                if (!recorded.count(Lower(FullPath(f))) && std::none_of(unrecorded.begin(), unrecorded.end(), [&](const wstring& u) { return IEq(FullPath(u), FullPath(f)); }))
                    unrecorded.push_back(f);
    if (!unrecorded.empty()) {
        wstring names;
        for (size_t i = 0; i < unrecorded.size(); ++i) {
            LogLine(ctx, L"kept " + unrecorded[i] + L" (named like the older install's files, but its record does not list it, so it is not deleted)");
            if (i < 5) names += (names.empty() ? L"" : L", ") + BaseName(unrecorded[i]);
        }
        if (unrecorded.size() > 5) names += L" and " + std::to_wstring(unrecorded.size() - 5) + L" more";
        warnings.push_back(kLeftUnrecorded + names);
    }
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
    if (!IEq(m.runtimeSha, cur.runtimeSha) || !IEq(m.frontEndSha, cur.frontEndSha) || !IEq(m.pluginSha, cur.pluginSha) ||
        !IEq(InstalledRrPluginSha(m), cur.rrPluginSha))
        return true;
    // frame generation's plugin (fg.h): where the install placed it and this build has it, by the record entry's hash (the
    // payload line has no field of its own for it); another build of it is an update too
    if (RecordHasFg(m) && FgPayloadAvailable(ctx)) {
        wstring installed, now;
        for (auto& e : m.e) if (e.action == L"copy" && IsFgPlugin(e.rel)) installed = e.inst;
        for (size_t i = 0; i < ctx.data->fgCount; ++i) if (IEq(ctx.data->fgFiles[i].rel, kFgPayloadPlugin)) now = ctx.data->fgFiles[i].sha256;
        if (!IEq(installed, now)) return true;
    }
    return false;
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
    {  // the 0.5.x record kept beside it, created only where nothing is (a file an earlier uninstall kept there stays)
        bool kept = false;
        for (int n = 0; n < 100 && !kept; ++n) kept = CopyNew(RecordPath(dir), RecordPath(dir) + L".v1" + (n ? L"." + std::to_wstring(n) : wstring()));
        if (!kept) Fail(L"Cannot keep a copy of the earlier record " + RecordPath(dir));
    }
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
    if (m.runtime.empty()) SetRuntimeLines(m, dir, m.exe.empty() ? dir : Parent(m.exe));  // the 0.5.x install's own wronggpu-* files
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