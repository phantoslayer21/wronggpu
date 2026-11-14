// SPDX-License-Identifier: MIT
// WrongGPU launcher core: data locations, global settings, the game list, the per-game operations, the elevated
// worker's job file, the launcher's own install, Play and the opt-in update check (launcher.h). Written for this
// project.
#include "launcher.h"

#include <objbase.h>
#include <sddl.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <winhttp.h>

#include <ctime>

#include "gamerules.h"
#include "json_writer.h"
#include "optiscaler.h"
#include "payload.h"
#include "pe.h"
#include "rr.h"
#include "rrconfig.h"

#ifdef _MSC_VER
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "uuid.lib")
#endif

namespace wg {

static string LowerA(string s) { for (auto& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a'); return s; }
static string TrimA(string t) {
    while (!t.empty() && (t.back() == '\r' || t.back() == '\n' || t.back() == ' ' || t.back() == '\t')) t.pop_back();
    size_t i = 0; while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
    return t.substr(i);
}
// key=value lines of a UTF-8 text (comments and section lines skipped), in order.
static std::vector<std::pair<string, string>> KeyValues(const string& text) {
    std::vector<std::pair<string, string>> r;
    for (size_t a = 0; a < text.size();) {
        size_t e = text.find('\n', a);
        if (e == string::npos) e = text.size();
        const string t = TrimA(text.substr(a, e - a));
        a = e + 1;
        if (t.empty() || t[0] == ';' || t[0] == '#' || t[0] == '[') continue;
        const size_t eq = t.find('=');
        if (eq == string::npos) continue;
        r.push_back({LowerA(TrimA(t.substr(0, eq))), TrimA(t.substr(eq + 1))});
    }
    return r;
}
static bool Truthy(const string& v) { const string l = LowerA(v); return l == "1" || l == "true" || l == "on" || l == "yes"; }

// ---- data locations ----------------------------------------------------------------------------------------------
wstring DefaultInstallDir() {
    const wstring base = EnvVar(L"LOCALAPPDATA");
    return Join(Join(base, L"Programs"), L"WrongGPU");
}
LauncherPaths ResolvePaths(const wstring& selfExe, const wstring& dataOverride) {
    LauncherPaths p;
    p.exe = selfExe;
    const wstring dir = Parent(selfExe);
    p.installedCopy = IEq(FullPath(dir), FullPath(DefaultInstallDir()));
    p.portable = !p.installedCopy && IsFile(Join(dir, kPortableMarker));
    if (!dataOverride.empty()) { p.custom = true; p.dataDir = FullPath(dataOverride); }
    else if (p.portable) p.dataDir = Join(dir, L"WrongGPU-data");
    else p.dataDir = Join(EnvVar(L"LOCALAPPDATA"), L"WrongGPU");
    return p;
}
bool RunningPackaged() {
    using Fn = LONG(WINAPI*)(UINT32*, PWSTR);
    static const Fn fn = (Fn)(void*)GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetCurrentPackageFullName");
    if (!fn) return false;
    UINT32 len = 0;
    return fn(&len, nullptr) == ERROR_INSUFFICIENT_BUFFER;
}
bool LauncherInstalled(wstring* exe) {
    const wstring key = UninstallRoot() + kLauncherEntry;
    const wstring loc = RegString(UninstallHive(), key.c_str(), L"InstallLocation");
    // an entry whose program is gone (its folder deleted by hand) is no install: Install for this user writes it again
    if (loc.empty() || !IsFile(Join(loc, L"WrongGPU.exe"))) return false;
    if (exe) *exe = Join(loc, L"WrongGPU.exe");
    return true;
}
bool UpdateDue(const GameEntry& e, const Context& ctx) {
    return e.hasRecord && !e.installedVersion.empty() && ctx.data && (VersionLess(e.installedVersion, ctx.data->version) || e.otherBuild);
}
wstring InstalledLabel(const GameEntry& e) {
    return e.otherBuild ? e.installedVersion + L" (another build)" : e.installedVersion;
}
wstring BuildTimeText(uint32_t t) {
    if (!t) return L"unknown";
    const time_t tt = (time_t)t;
    tm u{};
    if (gmtime_s(&u, &tt) != 0) return L"unknown";
    wchar_t b[40];
    swprintf(b, 40, L"%04d-%02d-%02d %02d:%02d UTC", u.tm_year + 1900, u.tm_mon + 1, u.tm_mday, u.tm_hour, u.tm_min);
    return b;
}
LauncherStartDecision DecideLauncherStart(const wstring& selfExe, const wstring& selfVersion, const wstring& installedExe,
                                          const wstring& installedVersion) {
    LauncherStartDecision d;
    d.selfBuilt = FileLinkTime(selfExe);
    d.installedBuilt = FileLinkTime(installedExe);
    const wstring builds = L" (this copy built " + BuildTimeText(d.selfBuilt) + L", the installed copy built " + BuildTimeText(d.installedBuilt) + L")";
    if (!d.installedBuilt) {  // no readable program there (corrupt or cut short): this copy runs, whatever the entry's version says
        d.choice = LauncherStart::RunHere;
        d.why = L"the installed copy isn't a readable program (its entry says " + (installedVersion.empty() ? wstring(L"no version") : installedVersion) + L")";
        return d;
    }
    if (VersionLess(installedVersion, selfVersion)) {
        d.choice = LauncherStart::OfferReplace;
        d.why = L"an older version is installed (" + installedVersion + L" < " + selfVersion + L")";
        return d;
    }
    if (VersionLess(selfVersion, installedVersion)) {
        d.why = L"a newer version is installed (" + installedVersion + L" > " + selfVersion + L")";
        return d;
    }
    d.sameVersion = true;
    // the same build: the same bytes (the size first, the hash only when the sizes agree)
    d.sameBuild = FileSize(selfExe) == FileSize(installedExe) && IEq(Sha256File(selfExe), Sha256File(installedExe));
    if (d.sameBuild) {
        d.why = L"the same build of " + selfVersion + L" is installed" + builds;
    } else if (d.installedBuilt > d.selfBuilt) {
        d.why = L"a newer build of " + selfVersion + L" is installed" + builds;
    } else {
        d.choice = LauncherStart::OfferReplace;
        d.why = L"an older build of " + selfVersion + L" is installed" + builds;
    }
    return d;
}
wstring CurrentUserSid() {
    HANDLE tok = nullptr;
    wstring r;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &tok)) return r;
    DWORD n = 0;
    GetTokenInformation(tok, TokenUser, nullptr, 0, &n);
    std::vector<BYTE> buf(n ? n : 1);
    LPWSTR s = nullptr;
    if (n && GetTokenInformation(tok, TokenUser, buf.data(), n, &n) && ConvertSidToStringSidW(((TOKEN_USER*)buf.data())->User.Sid, &s)) { r = s; LocalFree(s); }
    CloseHandle(tok);
    return r;
}

// ---- global settings ---------------------------------------------------------------------------------------------
static bool IsAccent(const string& a) {
    for (const char* k : {"green", "blue", "purple", "pink", "cyan", "graphite", "windows"}) if (a == k) return true;
    return false;
}
GlobalSettings LoadGlobal(const LauncherPaths& p) {
    GlobalSettings s;
    s.defaults = DefaultProfile();
    string raw;
    if (!ReadAll(Join(p.dataDir, L"launcher.ini"), raw)) return s;
    for (auto& kv : KeyValues(raw)) {
        const string& k = kv.first; const string& v = kv.second;
        if (k == "optiscaler") s.optiPackage = FromUtf8(v);
        else if (k == "extra_folder") { if (!v.empty()) s.extraFolders.push_back(FromUtf8(v)); }
        else if (k == "added_game") { if (!v.empty()) s.addedGames.push_back(FromUtf8(v)); }
        else if (k == "play_button") s.playButton = Truthy(v);
        else if (k == "update_check") s.updateCheck = Truthy(v);
        else if (k == "update_url") s.updateUrl = FromUtf8(v);
        else if (k == "appearance") { const string l = LowerA(TrimA(v)); if (l == "light" || l == "dark" || l == "system") s.appearance = FromUtf8(l); }
        else if (k == "accent") { const string l = LowerA(TrimA(v)); if (IsAccent(l)) s.accent = FromUtf8(l); }
        else if (k == "exact_mode") s.exactMode = Truthy(v);
        else if (k == "first_run_done") s.firstRunDone = Truthy(v);
        else if (k == "rr_first_start_noted") s.rrFirstStartNoted = Truthy(v);
        else if (k == "window") s.window = FromUtf8(TrimA(v));
        else if (k.rfind("defaults.", 0) == 0) SetSetting(s.defaults, k.substr(9), v, nullptr);
    }
    return s;
}
void SaveGlobal(const LauncherPaths& p, const GlobalSettings& s) {
    string t = "; WrongGPU launcher: global settings. Per-game settings live next to each game (wronggpu-setup\\profile.ini).\r\n[launcher]\r\n";
    t += "optiscaler=" + ToUtf8(s.optiPackage) + "\r\n";
    for (auto& f : s.extraFolders) t += "extra_folder=" + ToUtf8(f) + "\r\n";
    for (auto& g : s.addedGames) t += "added_game=" + ToUtf8(g) + "\r\n";
    t += string("play_button=") + (s.playButton ? "1" : "0") + "\r\n";
    t += string("; Opt-in: look for a new WrongGPU release at start (no network at all while this is 0 or the URL is empty).\r\n");
    t += string("update_check=") + (s.updateCheck ? "1" : "0") + "\r\n";
    t += "update_url=" + ToUtf8(s.updateUrl) + "\r\n";
    t += "; The window's theme: dark, light or system (follows Windows' app mode).\r\n";
    t += "appearance=" + ToUtf8(s.appearance) + "\r\n";
    t += "; The controls' colour: green, blue, purple, pink, cyan, graphite or windows (follows the Windows accent colour).\r\n";
    t += "accent=" + ToUtf8(s.accent) + "\r\n";
    t += "; Ray Reconstruction quality offers Exact too (the most precise mode, and the slowest). Off changes no game's setting.\r\n";
    t += string("exact_mode=") + (s.exactMode ? "1" : "0") + "\r\n";
    t += string("first_run_done=") + (s.firstRunDone ? "1" : "0") + "\r\n";
    t += string("rr_first_start_noted=") + (s.rrFirstStartNoted ? "1" : "0") + "\r\n";
    if (!s.window.empty()) t += "window=" + ToUtf8(s.window) + "\r\n";  // the window's place, kept as it closes
    for (size_t i = 0; i < kSettingCount; ++i)
        if (!s.defaults.isDefault(kSettings[i].key)) t += string("defaults.") + kSettings[i].key + "=" + s.defaults.get(kSettings[i].key) + "\r\n";
    EnsureDir(p.dataDir);
    WriteFileAtomic(Join(p.dataDir, L"launcher.ini"), t.data(), t.size());
}
bool SetGlobal(GlobalSettings& s, const string& keyIn, const string& value, wstring* why) {
    const string k = LowerA(TrimA(keyIn));
    if (k == "optiscaler") { s.optiPackage = value.empty() ? wstring() : FullPath(FromUtf8(value)); return true; }
    if (k == "extra_folder") { if (!value.empty()) s.extraFolders.push_back(FullPath(FromUtf8(value))); return true; }
    if (k == "added_game") { if (!value.empty()) s.addedGames.push_back(FullPath(FromUtf8(value))); return true; }
    if (k == "play_button" || k == "update_check" || k == "exact_mode") {
        const string l = LowerA(value);
        if (l != "0" && l != "1" && l != "on" && l != "off" && l != "true" && l != "false") { if (why) *why = L"use 1 or 0"; return false; }
        (k == "play_button" ? s.playButton : k == "update_check" ? s.updateCheck : s.exactMode) = Truthy(value);
        return true;
    }
    if (k == "update_url") {
        if (!value.empty() && LowerA(value).rfind("https://", 0) != 0) { if (why) *why = L"The release address must start with https://."; return false; }
        s.updateUrl = FromUtf8(value);
        return true;
    }
    if (k == "appearance") {
        const string l = LowerA(TrimA(value));
        if (l != "system" && l != "light" && l != "dark") { if (why) *why = L"use system, light or dark"; return false; }
        s.appearance = FromUtf8(l);
        return true;
    }
    if (k == "accent") {
        const string l = LowerA(TrimA(value));
        if (!IsAccent(l)) { if (why) *why = L"use green, blue, purple, pink, cyan, graphite or windows"; return false; }
        s.accent = FromUtf8(l);
        return true;
    }
    if (k.rfind("defaults.", 0) == 0) return SetSetting(s.defaults, k.substr(9), value, why);
    if (why) *why = L"unknown global setting \"" + FromUtf8(keyIn) + L"\"";
    return false;
}

// ---- game list -----------------------------------------------------------------------------------------------------
const wchar_t* StatusName(GameStatus s) {
    switch (s) {
        case GameStatus::Eligible: return L"eligible";
        case GameStatus::Installed: return L"installed";
        case GameStatus::UpdateAvailable: return L"update-available";
        case GameStatus::NeedsRepair: return L"needs-repair";
        case GameStatus::TurnedOff: return L"turned-off";
        case GameStatus::PlayersOptiScaler: return L"players-optiscaler";
        case GameStatus::ForeignMod: return L"foreign-mod";
        case GameStatus::OldInstall: return L"old-install";
        case GameStatus::Orphaned: return L"orphaned";
        case GameStatus::Refused: return L"refused";
        case GameStatus::Running: return L"running";
        case GameStatus::Leftovers: return L"leftovers";
        case GameStatus::NeedsPrereq: return L"needs-prerequisite";
    }
    return L"?";
}


// The record's problems: VerifyInstall (every file hashed), or with `quick` only whether each file is there (the start-up
// list's first look: ScanGames' managedFirst; DeepCheck hashes them afterwards). A missing settings file is one problem:
// the record lists wronggpu.ini as a file of the install too ("missing: wronggpu.ini"), and it is said once, in its own
// words (the count of files to repair stays right).
static std::vector<wstring> RecordProblems(const wstring& dir, const Record& rec, bool quick) {
    std::vector<wstring> problems;
    if (!quick) problems = VerifyInstall(dir, rec);
    else
        for (auto& e : rec.e) {
            if ((e.action != L"copy" && e.action != L"rename") || e.inst.empty() || IsMirrorRel(e.rel)) continue;
            if (!IsFile(Join(dir, e.rel))) problems.push_back(L"missing: " + e.rel);
        }
    problems.erase(std::remove_if(problems.begin(), problems.end(), [](const wstring& p) { return IEq(p, L"missing: wronggpu.ini"); }), problems.end());
    if (!IsFile(Join(dir, L"wronggpu.ini"))) problems.push_back(L"wronggpu.ini is missing");
    return problems;
}

wstring MisplacedFor(const wstring& exe) {
    if (exe.empty() || !IsFile(exe)) return L"";
    const wstring real = ResolveGameExe(exe);
    return !real.empty() && !IEq(FullPath(real), FullPath(exe)) && !IsLauncherStub(real) ? real : wstring();
}
// The problem of an install next to a stub or a starter (rc6 L2): first in the list, so the status says it.
static wstring MisplacedProblem(const wstring& exe, const wstring& real) {
    const wstring base = Parent(exe) + L"\\";
    const wstring rel = StartsWithI(real, base) ? real.substr(base.size()) : real;
    return wstring(kMisplacedProblem) + BaseName(exe) + (IsLauncherStub(exe) ? L", the game's launcher," : L", the game's starter,") +
           L" not the game's program " + rel + L": Repair moves it there";
}

GameEntry EntryFor(Context& ctx, const wstring& exeIn, const wstring& name, const wstring& source, bool quick) {
    GameEntry g;
    // rc6 L2: a stub or a starter with no install next to it is the game it leads to (the entry of an install that Repair just
    // moved to the game, read again)
    wstring exe = exeIn;
    if (!IsFile(RecordPath(Parent(exe))))
        if (const wstring real = MisplacedFor(exe); !real.empty()) exe = real;
    g.exe = exe; g.source = source;
    const bool exists = IsFile(exe);
    g.exeFound = exists;
    GameTarget t;
    if (exists) t = InspectGame(ctx, exe, name);
    g.name = !t.name.empty() ? t.name : !name.empty() ? name : GameDisplayName(exe);
    if (exists) g.appid = SteamAppId(exe);
    g.dir = exists ? t.dir : Parent(exe);
    g.coexist = t.coexist.findings;
    g.proxyPlan = t.coexist.proxy;
    for (auto& f : g.coexist) if (f.decision != Decision::Managed) g.found.push_back(f.what);
    if (exists && !quick) { const RrUse ru = DetectRr(exe); g.rrUses = ru.uses; g.rrWhy = ru.why; g.rrDirect = ru.direct; g.rrNotOffered = ru.notOffered; }  // quick: DeepCheck walks the folder
    if (const GameRule* rule = exists ? GameRuleFor(exe) : nullptr) {  // what the game needs beyond the usual (gamerules.h)
        g.rrPathTracingOnly = rule->rrOnlyPathTracing;
        g.needsGame = rule->name;
        if (rule->needsReFramework) {
            const ReFrameworkState refw = DetectReFramework(Parent(exe));
            g.needs = kReFrameworkName; g.needsPage = kReFrameworkPage; g.needsFound = refw.found; g.needsWhy = refw.evidence;
        }
    }
    if (t.installed) {
        const Record& rec = t.record;
        g.hasRecord = true;
        g.installedVersion = rec.version;
        // rc5: the same version holding another build's files (the payload line's hashes, or none): an update as well
        g.otherBuild = ctx.data && !rec.version.empty() && !VersionLess(rec.version, ctx.data->version) &&
                       !VersionLess(ctx.data->version, rec.version) && PayloadDiffers(ctx, rec);
        g.recordV1 = rec.format < 2;
        g.perGameEntry = !rec.regkey.empty();
        g.hosted = HostedRecord(rec);
        g.proxy = rec.proxy;
        g.optiEdition = RecordHasEdition(rec);
        if (g.proxy.empty())
            for (auto& e : rec.e) for (auto px : kOptiProxies) if (e.action == L"copy" && IEq(e.rel, px)) g.proxy = px;
        g.adoptable = g.recordV1 || !IsFile(ProfilePath(g.dir)) || (g.perGameEntry && LauncherInstalled());
        g.problems = RecordProblems(g.dir, rec, quick);
        g.realExe = MisplacedFor(exe);  // rc6 L2: an earlier build's install next to a stub or a starter
        if (!g.realExe.empty()) g.problems.insert(g.problems.begin(), MisplacedProblem(exe, g.realExe));
        // another build of this version (a 0.6.0 beta or rc): no repair problem but an update (otherBuild, UpdateDue; rc5)
        g.repairSuggested = !g.problems.empty();
        g.hasPrevious = PreviousFits(g.dir, rec, &g.previous);
        g.last = ReadLastRun(g.dir);
        if (RecordHasRr(rec)) {
            g.rrInstalled = g.rrUses = true;
            if (g.rrWhy.empty()) g.rrWhy = L"installed";
            const RrFileState rs = ReadRrSettings(g.dir);
            g.rrEnabled = rs.enabled;
            g.rrMode = FromUtf8(rs.mode);
        }
        // rc5 (gamerules.h): a game whose menu has no RR option is never an RR game here, even with an older install's RR
        // files (rrInstalled stays as recorded: a repair or update takes them out); a game whose RR waits for its
        // experimental opt-in holds it off while the setting is off
        if (g.rrNotOffered) {  // rc6: not offered by the game's files (the updated Hogwarts Legacy, rr.cpp), as a rule's line below
            const RrUse ru = DetectRr(exe);
            g.rrUses = false;
            if (!ru.why.empty()) g.rrWhy = ru.why;
        }
        if (const GameRule* rule = GameRuleFor(exe)) {
            if (rule->rrNotOffered) { g.rrNotOffered = true; g.rrUses = false; g.rrWhy = rule->rrNotOffered; }
            if (rule->rrExperimentalSetting && g.rrUses) {
                Profile p;
                g.rrHeldOff = !LoadProfile(ProfilePath(g.dir), p) || p.get(rule->rrExperimentalSetting) != "on";
            }
        }
        if (RrConfigGameFor(exe) && !rec.rrconfig.present) {  // "Fix it for me" chosen before the game's first start (r2b #1)
            Profile p;
            const bool loaded = LoadProfile(ProfilePath(g.dir), p);
            const string fix = loaded ? p.get("rr_config_fix") : string("ask");
            const int waiting = loaded ? RrConfigWaiting(ctx, rec, fix) : 0;
            // (only with WrongGPU's RR files, r7 #5: without them the line serves nothing of WrongGPU's, so the window never adds it by
            // itself; the player's own Fix it now, Apply, Repair and Update still can)
            g.rrConfigDue = waiting == 1 && g.rrInstalled && !g.rrNotOffered;  // (rc6: never where WrongGPU's RR isn't offered)
            // (RR's own switch decides whether the line needs the player, r5 #5: their rr_enabled, read above with WrongGPU's RR
            // files; an install without them never needs the player for it, r6 #5: RrConfigReadyToAdd)
            // whether adding it needs administrator permission: the window then waits for the player's click (r2c); also before
            // the choice is saved, so the page says so for an unsaved WrongGPU too (r3 #4). A write test in the game's folder at
            // every scan: only where WrongGPU may add the line (on, or not decided yet: the question), never for a player who
            // chose Myself (r4 #10)
            if (fix == "on" || fix == "ask") try { g.rrConfigProtected = NeedsElevation(g.dir); } catch (...) {}
        }
        string wini, oini;
        ReadAll(Join(g.dir, L"wronggpu.ini"), wini);
        ReadAll(Join(g.dir, L"OptiScaler.ini"), oini);
        const bool off = EffectiveSettings(wini, oini)["wronggpu"] == "off";
        g.wgOff = off;  // kept with a Running status too (the window's Model M data rule)
        if (GameRunning(exe)) { g.status = GameStatus::Running; g.statusText = L"Running (actions locked)"; }
        else if (!g.problems.empty()) { g.status = GameStatus::NeedsRepair; g.statusText = L"Needs repair (" + g.problems[0] + L")"; }
        else if (UpdateDue(g, ctx)) { g.status = GameStatus::UpdateAvailable; g.statusText = L"Installed " + InstalledLabel(g) + L" \u2192 update " + ctx.data->version; }
        // the player's tool the game needs is gone since the install (REFramework removed): the game blocks OptiScaler, so
        // WrongGPU can't run there; never "Installed" (healthy) while that is so. A repair or an update still comes first:
        // each is an action on WrongGPU's own files, and the page says what the game needs as well
        else if (!g.needs.empty() && !g.needsFound) { g.status = GameStatus::NeedsPrereq; g.statusText = L"Installed " + rec.version + L", needs " + g.needs + L": " + g.needsGame + L" blocks OptiScaler without it"; }
        else if (off) { g.status = GameStatus::TurnedOff; g.statusText = L"Installed " + rec.version + L", turned off (AMD FSR)"; }
        else { g.status = GameStatus::Installed; g.statusText = L"Installed " + rec.version; }
        return g;
    }
    if (!exists) {  // a managed install whose game is gone (moved or uninstalled): Uninstall removes our files
        Record rec;
        if (LoadRecord(RecordPath(g.dir), rec) && !rec.legacy) {
            g.hasRecord = true; g.installedVersion = rec.version; g.recordV1 = rec.format < 2; g.perGameEntry = !rec.regkey.empty(); g.hosted = HostedRecord(rec);
        }
        g.status = GameStatus::Orphaned;
        g.statusText = L"Orphaned: the game is not at " + exe + L" any more (Uninstall removes WrongGPU's files)";
        return g;
    }
    const wstring fsrDir = FindInstallDir(exe);
    Record direct;
    if (!IEq(fsrDir, g.dir) && LoadRecord(RecordPath(fsrDir), direct) && !direct.legacy) {
        g.status = GameStatus::OldInstall; g.installedVersion = direct.version;
        g.statusText = L"WrongGPU " + direct.version + L" in the FSR folder (an older layout): Install moves it to the OptiScaler setup";
    } else if (!t.antiCheat.empty()) { g.status = GameStatus::Refused; g.statusText = L"Refused: anti-cheat (" + ShortenMiddle(t.antiCheat, 60) + L")"; }
    else if (t.coexist.has(Tool::NrOnAmd)) { g.status = GameStatus::ForeignMod; g.statusText = L"Other upscaler mod: not touched"; }
    else if (const Finding* no = t.coexist.refusal()) { g.status = GameStatus::Refused; g.statusText = L"Refused: " + no->what; }
    else if (!g.needs.empty() && !g.needsFound) { g.status = GameStatus::NeedsPrereq; g.statusText = L"Needs " + g.needs + L": " + g.needsGame + L" blocks OptiScaler without it"; }
    else if (t.conflicts.oldFound()) { g.status = GameStatus::OldInstall; g.statusText = L"Has an older WrongGPU (DLSS 4.5 AMD): Install removes it first"; }
    else if (!(g.leftovers = FindLeftovers(ctx, g.dir)).empty() && !(g.leftoversOtherSetup = LeftoversUsedByOtherSetup(ctx, g.dir, g.leftovers))) {
        g.status = GameStatus::Leftovers;
        g.statusText = L"Leftover WrongGPU files without a record: " + LeftoverSummary(g.leftovers) + L" - Clean up or Adopt";
    }
    else if (t.coexist.needsSwitch()) { g.status = GameStatus::PlayersOptiScaler; g.statusText = L"Has your own OptiScaler: Install offers to switch to WrongGPU's OptiScaler setup"; }
    else if (!t.conflicts.mods.empty()) { g.status = GameStatus::ForeignMod; g.statusText = L"Other upscaler mod: " + t.conflicts.mods[0]; }
    else if (GameRunning(exe)) { g.status = GameStatus::Running; g.statusText = L"Running (actions locked)"; }
    else { g.status = GameStatus::Eligible; g.statusText = t.validated ? L"Ready to install (tested)" : L"Ready to install (experimental)"; }
    return g;
}

// What a quick entry (EntryFor's `quick`) left out, done now: the installed files hashed (a changed file makes it Needs
// repair, as EntryFor would have) and the Ray Reconstruction folder walk. Everything else of the entry stays.
static void DeepCheck(GameEntry& g) {
    if (g.exeFound) {
        const RrUse ru = DetectRr(g.exe);
        g.rrUses = ru.uses || g.rrInstalled;
        g.rrWhy = !ru.why.empty() ? ru.why : g.rrInstalled ? wstring(L"installed") : wstring();
        g.rrDirect = ru.direct;
        if (ru.notOffered) { g.rrNotOffered = true; g.rrUses = false; g.rrConfigDue = false; g.rrHeldOff = false; }  // rc6 (rr.cpp)
    }
    if (!g.hasRecord || !g.exeFound) return;  // a game that is gone has nothing to hash
    Record rec;
    if (!LoadRecord(RecordPath(g.dir), rec) || rec.legacy) return;
    g.problems = RecordProblems(g.dir, rec, false);  // the quick look's missing files are among these
    if (!g.realExe.empty()) g.problems.insert(g.problems.begin(), MisplacedProblem(g.exe, g.realExe));  // rc6 L2
    g.repairSuggested = !g.problems.empty();
    if (!g.problems.empty() && g.status != GameStatus::Running) { g.status = GameStatus::NeedsRepair; g.statusText = L"Needs repair (" + g.problems[0] + L")"; }
}

static void SortGames(std::vector<GameEntry>& r) {
    std::stable_sort(r.begin(), r.end(), [](const GameEntry& a, const GameEntry& b) {
        if (a.hasRecord != b.hasRecord) return a.hasRecord;
        return Lower(a.name) < Lower(b.name);
    });
}

std::vector<GameEntry> ScanGames(Context& ctx, const GlobalSettings& gs, bool noScan, const std::function<void(const std::vector<GameEntry>&)>& managedFirst) {
    std::vector<GameEntry> r;
    std::set<wstring> seen;
    // the managed games are read quickly when they are shown first (managedFirst: their files checked for being there, not
    // hashed; no Ray Reconstruction folder walk), so the start-up list comes in a moment, then checked in full (DeepCheck)
    // before the store libraries are read
    const bool quick = (bool)managedFirst;
    auto add = [&](const wstring& exe, const wstring& name, const wstring& source, bool fast) {
        if (exe.empty() || seen.count(Lower(FullPath(exe)))) return;
        seen.insert(Lower(FullPath(exe)));
        // one game that can't be read (a broken or huge file, a parser's throw) is left out and logged, never the whole list
        try {
            r.push_back(EntryFor(ctx, FullPath(exe), name, source, fast));
            // the game of an install in the wrong place (rc6 L2) is that entry: never listed a second time by a store
            // (unless the game has an install of its own as well: then both are listed)
            if (const wstring& real = r.back().realExe; !real.empty() && !IsFile(RecordPath(Parent(real)))) seen.insert(Lower(FullPath(real)));
            if (!IEq(r.back().exe, FullPath(exe))) seen.insert(Lower(FullPath(r.back().exe)));  // a stub read as its game
        }
        catch (const SetupError& e) { LogLine(ctx, L"scan: " + FullPath(exe) + L" left out: " + e.msg); }
        catch (...) { LogLine(ctx, L"scan: " + FullPath(exe) + L" left out: it could not be read"); }
    };
    std::vector<wstring> gone;  // rc6 L1: managed folders without a record on a drive that is connected (moved or deleted)
    for (auto& dir : ManagedGames()) {
        Record m;
        if (!IsFile(RecordPath(dir)) && !DriveUnreachable(dir)) { gone.push_back(dir); continue; }
        if (!LoadRecord(RecordPath(dir), m) || m.legacy) continue;
        // a game that is gone: its entry points into the install folder, where the record is
        add(IsFile(m.exe) ? m.exe : Join(dir, m.exe.empty() ? wstring(L"game.exe") : BaseName(m.exe)), L"", L"Managed", quick);
    }
    for (auto& dir : gone) {  // off the list (a moved game comes back by its new folder: the store scan, its rebased record)
        try { RemoveManagedGame(dir); LogLine(ctx, L"scan: " + dir + L" left the managed list: its install record is gone"); }
        catch (const SetupError& e) { LogLine(ctx, L"scan: " + dir + L" could not leave the managed list: " + e.msg); }
    }
    if (managedFirst && !r.empty()) { std::vector<GameEntry> first = r; SortGames(first); managedFirst(first); }
    if (quick)  // a hash that throws keeps the quick entry (logged), as a game that can't be read is left out above
        for (auto& g : r) {
            try { DeepCheck(g); }
            catch (const SetupError& e) { LogLine(ctx, L"scan: " + g.exe + L" not checked in full: " + e.msg); }
            catch (...) { LogLine(ctx, L"scan: " + g.exe + L" not checked in full"); }
        }
    // then the store libraries and the extra folders (read after the managed games: the same list, sooner for them)
    std::vector<StoreGame> roots = noScan ? std::vector<StoreGame>{} : ScanLibraries();
    for (auto& folder : gs.extraFolders)
        for (auto& sub : ListDirs(folder)) roots.push_back({BaseName(sub), sub, L"", L"Folder"});
    for (auto& f : DetectGames(roots)) add(f.exe, f.name, f.source, false);
    // a game added by hand whose exe is gone (deleted, its drive offline) is not listed; it stays on the list for when it
    // is back (a managed one came in above, by its record)
    // (rc6: as the game it leads to, ResolveGameExe: a launcher stub or a root starter added before Add game resolved it)
    for (auto& exe : gs.addedGames)
        if (IsFile(exe)) { const wstring game = ResolveGameExe(exe); add(game.empty() ? exe : game, L"", L"Added", false); }
    SortGames(r);
    return r;
}

bool ResolveGameArg(Context& ctx, const wstring& arg, wstring& exe, wstring& dir) {
    exe.clear(); dir.clear();
    if (arg.empty()) return false;
    wstring in = arg;
    while (!in.empty() && (in.back() == L'\\' || in.back() == L'/')) in.pop_back();
    const wstring full = FullPath(in);
    Record m;
    if (IsDir(full) && LoadRecord(RecordPath(full), m) && !m.legacy) {
        dir = full;
        exe = IsFile(m.exe) ? m.exe : Join(full, m.exe.empty() ? wstring(L"game.exe") : BaseName(m.exe));  // gone: points into the folder
        return true;
    }
    exe = ResolveGameExe(full);
    if (exe.empty()) return false;
    GameTarget t = InspectGame(ctx, exe);
    dir = t.dir;
    return true;
}

// ---- operations ----------------------------------------------------------------------------------------------------
const wchar_t* OpName(Op o) {
    switch (o) {
        case Op::Install: return L"install";
        case Op::Repair: return L"repair";
        case Op::Update: return L"update";
        case Op::Uninstall: return L"uninstall";
        case Op::Rollback: return L"rollback";
        case Op::Apply: return L"apply";
        case Op::Adopt: return L"adopt";
        case Op::CleanLeftovers: return L"clean-leftovers";
        case Op::AdoptLeftovers: return L"adopt-leftovers";
        case Op::Detach: return L"detach";
    }
    return L"?";
}
bool ParseOp(const wstring& s, Op& o) {
    for (Op x : {Op::Install, Op::Repair, Op::Update, Op::Uninstall, Op::Rollback, Op::Apply, Op::Adopt, Op::CleanLeftovers, Op::AdoptLeftovers, Op::Detach})
        if (IEq(s, OpName(x))) { o = x; return true; }
    return false;
}
bool NeedsElevation(const wstring& dir) {
    if (IsElevated()) return false;
#ifdef WRONGGPU_TESTING
    // tests: the folders in WRONGGPU_TEST_PROTECTED (';'-separated; a folder in one of them too) count as protected, except in
    // the elevated worker's stand-in (WRONGGPU_TEST_ELEVATED=1, cli.cpp StartElevated) (r2c)
    if (EnvVar(L"WRONGGPU_TEST_ELEVATED") != L"1") {
        const wstring list = EnvVar(L"WRONGGPU_TEST_PROTECTED"), d = Lower(FullPath(dir));
        wstring cur;
        for (wchar_t c : list + L";") {
            if (c != L';') { cur += c; continue; }
            while (!cur.empty() && (cur.back() == L'\\' || cur.back() == L'/')) cur.pop_back();
            const wstring pfx = cur.empty() ? wstring() : Lower(FullPath(cur));
            if (!pfx.empty() && (d == pfx || (d.size() > pfx.size() && d.compare(0, pfx.size(), pfx) == 0 && d[pfx.size()] == L'\\'))) return true;
            cur.clear();
        }
    }
#endif
    const DWORD e = WriteTest(dir);
    return e == ERROR_ACCESS_DENIED || e == ERROR_WRITE_PROTECT;
}

static wstring PreviousDir(const wstring& dir) { return Join(StateDir(dir), L"previous"); }
bool HasPrevious(const wstring& dir, wstring* version) {
    string raw;
    if (!ReadAll(Join(PreviousDir(dir), L"previous.txt"), raw)) return false;
    for (auto& kv : KeyValues(raw)) if (kv.first == "version" && version) *version = FromUtf8(kv.second);
    return true;
}
// previous\: the payload files of the installed version (owner wg, unchanged), for Roll back. One version is kept.
static void SavePrevious(Context& ctx, const wstring& dir, Record m, Ui& ui) {
    InferOwners(ctx, m);
    const wstring prev = PreviousDir(dir), tmp = prev + L".new";
    RemoveTree(tmp);
    EnsureDir(tmp);
    string list = "; WrongGPU launcher: the version before the last update (Roll back puts it back)\r\nversion=" + ToUtf8(m.version) + "\r\n";
    size_t n = 0;
    for (auto& e : m.e) {
        if (e.action != L"copy" || e.owner != L"wg" || EndsWithI(e.rel, L".ini") || e.inst.empty()) continue;
        const wstring t = Target(dir, e.rel);
        if (!IsFile(t) || Sha256File(t) != e.inst) continue;
        CopyChecked(t, Join(tmp, e.rel));
        list += "file=" + ToUtf8(e.inst) + "|" + ToUtf8(e.rel) + "\r\n";
        ++n;
    }
    WriteFileAtomic(Join(tmp, L"previous.txt"), list.data(), list.size());
    RemoveTree(prev);
    if (!MoveFileExW(tmp.c_str(), prev.c_str(), MOVEFILE_WRITE_THROUGH)) Fail(L"Cannot keep the previous version: " + ErrText(GetLastError()));
    ui.log(L"kept WrongGPU " + m.version + L" for Roll back (" + std::to_wstring(n) + L" files in wronggpu-setup\\previous)");
}
struct PrevFile { wstring sha, rel; };
static bool ReadPrevious(const wstring& dir, wstring& version, std::vector<PrevFile>& files);
// The kept files must be files of the current install (same names): after the OptiScaler 10 route change the front end
// moved to OptiScaler's custom slot and the menu plugin is gone, so an earlier version's files can't go back in place.
static bool PreviousMatchesRecord(const std::vector<PrevFile>& files, const Record& m) {
    for (auto& f : files) {
        bool found = false;
        for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, f.rel)) { found = true; break; }
        if (!found) return false;
    }
    return true;
}
bool PreviousFits(const wstring& dir, const Record& m, wstring* version) {
    wstring v;
    std::vector<PrevFile> files;
    if (!ReadPrevious(dir, v, files) || files.empty() || !PreviousMatchesRecord(files, m)) return false;
    if (version) *version = v;
    return true;
}
static bool ReadPrevious(const wstring& dir, wstring& version, std::vector<PrevFile>& files) {
    string raw;
    if (!ReadAll(Join(PreviousDir(dir), L"previous.txt"), raw)) return false;
    for (auto& kv : KeyValues(raw)) {
        if (kv.first == "version") version = FromUtf8(kv.second);
        else if (kv.first == "file") {
            const size_t bar = kv.second.find('|');
            if (bar != string::npos) files.push_back({FromUtf8(kv.second.substr(0, bar)), FromUtf8(kv.second.substr(bar + 1))});
        }
    }
    return true;
}
static void Rollback(Context& ctx, const wstring& dir, Ui& ui, OpResult& res) {
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy) Fail(L"No WrongGPU installation record was found in " + dir + L".", EXIT_NOT_INSTALLED);
    if (!m.exe.empty() && GameRunning(m.exe)) Fail(BaseName(m.exe) + L" is running. Close it first.", EXIT_REFUSED);
    wstring version;
    std::vector<PrevFile> files;
    if (!ReadPrevious(dir, version, files) || files.empty()) Fail(L"There is no earlier version to roll back to in " + dir + L".", EXIT_NOT_INSTALLED);
    const wstring prev = PreviousDir(dir);
    if (!PreviousMatchesRecord(files, m))
        Fail(L"WrongGPU " + version + L" used a different OptiScaler setup, so it can't be put back here. Nothing was changed. Uninstall and install that version instead.", EXIT_REFUSED);
    for (auto& f : files) if (Sha256File(Join(prev, f.rel)) != f.sha) Fail(L"The kept earlier version is damaged (" + f.rel + L"). Nothing was changed.", EXIT_FAILED);
    InferOwners(ctx, m);
    // the current files become the new "previous" (Roll back twice = back to this version)
    const wstring swap = prev + L".swap";
    RemoveTree(swap);
    EnsureDir(swap);
    string list = "; WrongGPU launcher: the version before the last roll back\r\nversion=" + ToUtf8(m.version) + "\r\n";
    for (auto& f : files) {
        const wstring t = Target(dir, f.rel);
        if (!IsFile(t)) continue;
        const wstring h = Sha256File(t);
        CopyChecked(t, Join(swap, f.rel));
        list += "file=" + ToUtf8(h) + "|" + ToUtf8(f.rel) + "\r\n";
    }
    WriteFileAtomic(Join(swap, L"previous.txt"), list.data(), list.size());
    // the earlier files go in, then the record names their hashes
    for (auto& f : files) {
        const wstring t = Target(dir, f.rel);
        CopyChecked(Join(prev, f.rel), t);
        if (Sha256File(t) != f.sha) Fail(L"Verification failed after putting back " + f.rel);
        for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, f.rel)) e.inst = f.sha;
        if (IEq(f.rel, L"wronggpu_runtime.dll")) m.runtimeSha = f.sha;
        else if (IsOptiFrontEnd(f.rel) || IEq(f.rel, L"amd_fidelityfx_dx12.dll")) m.frontEndSha = f.sha;
        else if (IEq(f.rel, kAsiPlugin)) m.pluginSha = f.sha;
        else if (IsRrPlugin(f.rel)) m.rrPluginSha = f.sha;  // the payload line's RR plugin build (rc6)
        ui.log(L"rolled back " + f.rel);
    }
    const wstring newer = m.version;
    m.version = version; m.payloadVersion = version;
    SaveRecord(RecordPath(dir), m);
    WriteOwnedFiles(dir, m);
    RemoveTree(prev);
    if (!MoveFileExW(swap.c_str(), prev.c_str(), MOVEFILE_WRITE_THROUGH)) res.warnings.push_back(L"the newer version could not be kept for a later update: " + ErrText(GetLastError()));
    res.message = L"Rolled back from " + newer + L" to " + version + L" (" + std::to_wstring(files.size()) + (files.size() == 1 ? L" file)" : L" files)");
}

// The section in which `ini` has `key` (the first one; keys compared without case), as written in the file.
static bool IniKeySection(const string& ini, const char* key, string* section) {
    string current;
    size_t at = 0;
    while (at < ini.size()) {
        size_t end = ini.find('\n', at);
        if (end == string::npos) end = ini.size();
        string line = ini.substr(at, end - at);
        at = end + 1;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        const size_t b = line.find_first_not_of(" \t");
        if (b == string::npos) continue;
        line = line.substr(b);
        if (line[0] == ';' || line[0] == '#') continue;
        if (line[0] == '[') {
            const size_t c = line.find(']');
            current = line.substr(1, c == string::npos ? string::npos : c - 1);
            continue;
        }
        const size_t eq = line.find('=');
        if (eq == string::npos) continue;
        string k = line.substr(0, eq);
        while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
        if (_stricmp(k.c_str(), key) == 0) {
            *section = current;
            return true;
        }
    }
    return false;
}

// A game's experimental opt-in keys (optiscaler.h kOptiGameKeys) in its installed OptiScaler.ini `in`: setting on, each
// key the file has set to its value (the journal then keeps the install's value); off, each key whose value is still ours
// put back to the install's value from the journal (a value the player changed since, e.g. on the OptiScaler tab, stays).
// *wrote: some key was set to the opt-in's value now.
static string RenderGameOptiKeys(const Profile& p, const wstring& exe, const string& in, const std::vector<IniKeyChange>& journal,
                                 std::vector<wstring>& notes, bool* wrote) {
    string t = in;
    *wrote = false;
    for (auto& g : kOptiGameKeys) {
        if (!IEq(BaseName(exe), g.exe)) continue;
        const bool on = p.get(g.setting) == "on";
        string section = g.section ? g.section : "";
        if (!g.section && !IniKeySection(t, g.key, &section)) section.clear();
        const IniSpan s = section.empty() ? IniSpan{} : IniFind(t, section, g.key);
        if (!s.found) {
            if (on) notes.push_back(L"OptiScaler.ini: this OptiScaler build has no " + FromUtf8(g.key) + L" key; not set for " + FromUtf8(g.setting));
            continue;
        }
        const string cur = IniValueAt(t, s);
        if (on) {
            if (cur == g.value) continue;
            t = IniReplaceValue(t, s, g.value);
            *wrote = true;
            notes.push_back(L"OptiScaler.ini: [" + FromUtf8(section) + L"] " + FromUtf8(g.key) + L"=" + FromUtf8(g.value) + L" (" + FromUtf8(g.setting) + L"=on)");
            continue;
        }
        if (cur != g.value) continue;  // not ours (the install's value, or the player's)
        for (auto& c : journal)
            if (IEq(c.file, L"OptiScaler.ini") && !c.absent && _stricmp(c.section.c_str(), section.c_str()) == 0 &&
                _stricmp(c.key.c_str(), g.key) == 0 && c.ours == g.value) {
                t = IniReplaceValue(t, s, c.original);
                notes.push_back(L"OptiScaler.ini: [" + FromUtf8(section) + L"] " + FromUtf8(g.key) + L" back to " + FromUtf8(c.original) + L" (" +
                                FromUtf8(g.setting) + L"=off)");
                break;
            }
    }
    return t;
}
// The journal entries of a game's opt-in keys (kept out of the OptiScaler tab's reset and the update's replay: the setting
// owns them).
static bool OptInKey(const wstring& exe, const IniKeyChange& c) {
    for (auto& g : kOptiGameKeys)
        if (IEq(BaseName(exe), g.exe) && IEq(c.file, L"OptiScaler.ini") && _stricmp(c.key.c_str(), g.key) == 0 &&
            (!g.section || _stricmp(c.section.c_str(), g.section) == 0))
            return true;
    return false;
}

Profile GameProfile(const wstring& dir) {
    Profile p;
    if (!LoadProfile(ProfilePath(dir), p)) {
        string wini, oini;
        ReadAll(Join(dir, L"wronggpu.ini"), wini);
        ReadAll(Join(dir, L"OptiScaler.ini"), oini);
        p = ImportProfile(wini, oini);
    }
    FoldRrFromGame(dir, p);  // Ray Reconstruction's mode as the player last chose it in the game (OptiScaler's menu)
    return p;
}

void ApplyProfile(Context& ctx, const wstring& dir, const Profile& p, std::vector<wstring>& notes, std::vector<wstring>* warnings) {
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy) Fail(L"WrongGPU is not installed in " + dir + L".", EXIT_NOT_INSTALLED);
    InferOwners(ctx, m);
    bool recordChanged = m.format < 2;
    auto renderFile = [&](const wchar_t* name, bool opti) {
        const wstring path = Join(dir, name);
        string in;
        const bool had = ReadAll(path, in);
        if (opti && !had) return;  // OptiScaler.ini: only an existing file (ours) is changed
        string out; wstring why;
        if (opti) {
            RenderOptiIni(p, in, out, &notes);
            bool wrote = false;
            out = RenderGameOptiKeys(p, m.exe, out, m.inikeys, notes, &wrote);
            if (wrote)
                for (auto& n : kGameNotices) if (IEq(BaseName(m.exe), n.exe)) notes.push_back(n.notice);
            JournalChanges(m.inikeys, name, in, out);
            recordChanged = recordChanged || out != in;
        }
        else if (!RenderWronggpuIni(p, in, out, &why)) Fail(L"Settings not applied: " + why, EXIT_USAGE);
        if (out == in && (had || out.empty())) return;  // unchanged, or nothing to write into a missing file
        WriteFileAtomic(path, out.data(), out.size());
        const wstring h = Sha256Bytes(out.data(), out.size());
        for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, name) && e.inst != h) { e.inst = h; recordChanged = true; }
        notes.push_back(wstring(name) + L" written");
    };
    renderFile(L"wronggpu.ini", false);
    if (HostedRecord(m)) renderFile(L"OptiScaler.ini", true);
    if (RecordHasRr(m)) {  // Ray Reconstruction (rr.h): the RR plugin's settings file (RR stays off while its data is not ready)
        const wstring path = Join(dir, kRrSettings);
        string in;
        ReadAll(path, in);
        const string out = RenderRrSettings(p, RrDataValid(ctx), in);
        if (out != in) {
            EnsureDir(Parent(path));
            WriteFileAtomic(path, out.data(), out.size());
            const wstring h = Sha256Bytes(out.data(), out.size());
            for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, kRrSettings) && e.inst != h) { e.inst = h; recordChanged = true; }
            notes.push_back(wstring(kRrSettings) + L" written");
        }
    }
    if (RrConfigGameFor(m.exe)) {  // Borderlands 4, Hogwarts Legacy: the game's Engine.ini follows rr_config_fix (rrconfig.h)
        std::vector<wstring> w;
        if (p.get("rr_config_fix") == "on") recordChanged = ApplyRrConfig(ctx, dir, m, notes, w) || recordChanged;
        else if (m.rrconfig.present) { RevertRrConfig(ctx, dir, m, notes, w); recordChanged = true; }
        for (auto& x : w) { notes.push_back(x); if (warnings) warnings->push_back(x); }
    }
    SaveProfile(ProfilePath(dir), p);
    if (recordChanged) SaveRecord(RecordPath(dir), m);
    WriteOwnedFiles(dir, m);
    LogLine(ctx, L"settings applied to " + dir + L": " + FromUtf8(ProfileText(p).substr(ProfileText(p).find("format=1") + 10)));
}

void ApplyRrConfigOnly(Context& ctx, const wstring& dir, std::vector<wstring>& notes, std::vector<wstring>* warnings, const string& set) {
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy) Fail(L"WrongGPU is not installed in " + dir + L".", EXIT_NOT_INSTALLED);
    if (!RrConfigGameFor(m.exe)) Fail(L"The Engine.ini setting is for " + RrConfigGameExes() + L" only.", EXIT_USAGE);
    // the choice as saved (profile.ini; an install without one: the profile its files imply), never a value of the request
    Profile p;
    const bool hadProfile = LoadProfile(ProfilePath(dir), p);
    if (!hadProfile) p = GameProfile(dir);
    // the question's answer (r4 #4): that one value saved into the profile as it is now; the game's other settings files and
    // the profile's other values stay as they are (a change made since the question was asked is never put back)
    if (!set.empty()) {
        wstring why;
        if (!SetSetting(p, "rr_config_fix", set, &why)) Fail(L"rr_config_fix: " + why, EXIT_USAGE);
        SaveProfile(ProfilePath(dir), p);
        notes.push_back(L"rr_config_fix saved: " + FromUtf8(set));
        LogLine(ctx, L"rr_config_fix=" + FromUtf8(set) + L" saved in the profile of " + dir + L" (no other value or file written)");
    }
    const string fix = p.get("rr_config_fix");
    if (fix == "off" && !set.empty() && m.rrconfig.present) {  // Myself chosen: WrongGPU's own line out again, as an Apply does
        InferOwners(ctx, m);
        std::vector<wstring> w;
        RevertRrConfig(ctx, dir, m, notes, w);
        SaveRecord(RecordPath(dir), m);
        for (auto& x : w) { notes.push_back(x); if (warnings) warnings->push_back(x); }
        return;
    }
    if (fix != "on") {
        notes.push_back(L"Engine.ini left as it is: rr_config_fix is " + FromUtf8(fix) + L" in the saved profile");
        LogLine(ctx, L"rr_config_fix: " + FromUtf8(fix) + L" in the saved profile of " + dir + L": Engine.ini left as it is");
        return;
    }
    InferOwners(ctx, m);
    std::vector<wstring> w;
    if (ApplyRrConfig(ctx, dir, m, notes, w)) SaveRecord(RecordPath(dir), m);
    for (auto& x : w) { notes.push_back(x); if (warnings) warnings->push_back(x); }
    LogLine(ctx, L"rr_config_fix: Engine.ini applied for " + dir + L" (no other file written)");
}

void ApplyOptiTab(Context& ctx, const wstring& dir, const OpRequest& r, std::vector<wstring>& notes) {
    if (r.optiEdits.empty() && !r.optiReset && !r.optiKeep) return;
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy) Fail(L"WrongGPU is not installed in " + dir + L".", EXIT_NOT_INSTALLED);
    if (!HostedRecord(m)) Fail(L"WrongGPU's OptiScaler setup is not installed in " + dir + L": there are no OptiScaler settings to change.", EXIT_NOT_INSTALLED);
    InferOwners(ctx, m);
    const wchar_t* const file = L"OptiScaler.ini";
    const wstring path = Join(dir, file);
    string in;
    if (!ReadAll(path, in)) Fail(L"OptiScaler.ini is missing in " + dir + L": Repair puts it back.", EXIT_FAILED);
    string text = in;
    if (r.optiReset) {  // the editable keys back to what the install wrote, byte for byte (locked keys stay as they are)
        std::vector<IniKeyChange> mine, rest;
        for (auto& c : m.inikeys) (IEq(c.file, file) && !IsOptiLocked(c.section, c.key) && !OptInKey(m.exe, c) ? mine : rest).push_back(c);
        text = RevertJournal(text, mine, file, &notes);
        m.inikeys = rest;
        notes.push_back(std::to_wstring(mine.size()) + L" OptiScaler setting(s) put back to what the install wrote");
    }
    if (r.optiKeep) {  // values saved over ours (OptiScaler's menu): they become the launcher's (the tab's keys: a locked one is
        // WrongGPU's, put back by its profile or a repair, never adopted here)
        for (auto& o : JournalOutside(text, m.inikeys, file))
            for (auto& c : m.inikeys)
                if (IEq(c.file, file) && c.section == o.section && c.key == o.key && o.fileValue != "(missing)" && !IsOptiLocked(o.section, o.key)) {
                    c.ours = o.fileValue;
                    notes.push_back(L"kept [" + FromUtf8(o.section) + L"] " + FromUtf8(o.key) + L" = " + FromUtf8(o.fileValue));
                }
        m.inikeys.erase(std::remove_if(m.inikeys.begin(), m.inikeys.end(), [&](const IniKeyChange& c) { return IEq(c.file, file) && !c.absent && c.original == c.ours; }),
                        m.inikeys.end());
    }
    if (!r.optiEdits.empty()) {
        string out; wstring why;
        if (!ApplyOptiEdits(text, r.optiEdits, out, &why)) Fail(L"OptiScaler settings not applied: " + why, EXIT_USAGE);
        JournalChanges(m.inikeys, file, text, out);
        for (auto& e : r.optiEdits) notes.push_back(L"OptiScaler " + FromUtf8(e.first) + L" = " + FromUtf8(e.second));
        text = out;
    }
    if (text != in) {
        WriteFileAtomic(path, text.data(), text.size());
        notes.push_back(L"OptiScaler.ini written");
    }
    const wstring h = Sha256Bytes(text.data(), text.size());
    for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, file)) e.inst = h;
    SaveRecord(RecordPath(dir), m);
    WriteOwnedFiles(dir, m);
    LogLine(ctx, L"OptiScaler tab applied to " + dir + L" (" + std::to_wstring(r.optiEdits.size()) + L" edit(s)" + (r.optiReset ? L", reset" : L"") + (r.optiKeep ? L", keep" : L"") + L")");
}

std::vector<wstring> AdoptInstall(Context& ctx, const wstring& dir, bool* repair) {
    std::vector<wstring> notes;
    Record m;
    if (!LoadRecord(RecordPath(dir), m) || m.legacy) Fail(L"No WrongGPU installation record was found in " + dir + L".", EXIT_NOT_INSTALLED);
    UpgradeRecord(ctx, dir, &notes);
    LoadRecord(RecordPath(dir), m);
    InferOwners(ctx, m);
    if (!m.regkey.empty() && LauncherInstalled()) {
        // the record first: a folder that refuses the write keeps its entry and a record that still names it (never a game
        // without any entry)
        Record without = m;
        without.regkey.clear();
        SaveRecord(RecordPath(dir), without);
        UnregisterUninstall(m);
        m.regkey.clear();
        notes.push_back(L"its own Apps & features entry was replaced by the launcher's (WrongGPU)");
    }
    if (!IsFile(ProfilePath(dir))) {
        string wini, oini;
        ReadAll(Join(dir, L"wronggpu.ini"), wini);
        ReadAll(Join(dir, L"OptiScaler.ini"), oini);
        std::vector<wstring> pn;
        const Profile p = ImportProfile(wini, oini, &pn);
        SaveProfile(ProfilePath(dir), p);
        notes.push_back(L"settings imported from wronggpu.ini into profile.ini (only the launcher's settings)");
        for (auto& n : pn) {
            notes.push_back(n);  // "kept your setting ...", "not imported: ...", "ignored: key=value (...)"
            LogLine(ctx, L"adopt " + dir + L": " + n);
        }
    }
    if (ctx.restorer && ctx.restorerSize) {
        const wstring un = Join(StateDir(dir), L"uninstall.exe");
        if (Sha256File(un) != Sha256Bytes(ctx.restorer, ctx.restorerSize)) {
            WriteFileAtomic(un, ctx.restorer, ctx.restorerSize);
            notes.push_back(L"uninstall.exe is now the launcher's restorer (reads the v1 and v2 records)");
        }
    }
    WriteOwnedFiles(dir, m);
    AddManagedGame(dir);
    const bool rep = !IsFile(Join(dir, L"wronggpu.ini")) || (HostedRecord(m) && RecordCopies(m, kAsiPlugin) && !IsFile(Join(dir, kAsiPlugin))) || !VerifyInstall(dir, m).empty();
    if (rep) notes.push_back(L"repair suggested (a file is missing or changed: wronggpu.ini, the plugin or a payload file)");
    if (repair) *repair = rep;
    LogLine(ctx, L"adopted the install in " + dir);
    return notes;
}

static void InstallOp(Context& ctx, const OpRequest& r, Ui& ui, OpResult& res) {
    if (r.exe.empty() || !IsFile(r.exe)) Fail(L"No game executable found. Name the game's .exe or folder.", EXIT_REFUSED);
    const wstring pkg = r.optiPackage;
    if (!pkg.empty() && (!ctx.opti.ok() || !IEq(ctx.opti.source, FullPath(pkg)))) {
        ctx.opti = PrepareOptiScaler(ctx, pkg);
        if (!ctx.opti.ok()) LogLine(ctx, L"OptiScaler: " + ctx.opti.error);
    }
    RequireOptiScaler(ctx);
    // rc6 L2: never next to a launcher stub or a starter: the game it leads to. An earlier build's install next to it (or in
    // the game's root, when the game itself is named) is moved there: uninstalled where it is, its settings carried over
    wstring gameExe = r.exe, wrongDir;
    if (const wstring real = MisplacedFor(r.exe); !real.empty()) {
        if (IsFile(RecordPath(Parent(r.exe)))) wrongDir = Parent(r.exe);
        LogLine(ctx, L"launcher: " + BaseName(r.exe) + L" is not the game's program: " + real + L" is");
        gameExe = real;
    } else if (const wstring root = GameRoot(Parent(r.exe)); !IEq(FullPath(root), FullPath(Parent(r.exe)))) {
        Record w;
        if (LoadRecord(RecordPath(root), w) && !w.legacy && !w.exe.empty() && IEq(FullPath(MisplacedFor(w.exe)), FullPath(r.exe))) wrongDir = root;
    }
    if (IsLauncherStub(gameExe)) Fail(BaseName(gameExe) + L" starts the game but isn't the game's program, and the game's program was not found next to it. Name the game's .exe.", EXIT_REFUSED);
    GameTarget target = InspectGame(ctx, gameExe);
    const wstring dir = target.dir;
    LogLine(ctx, wstring(L"launcher ") + OpName(r.op) + L": " + target.exe + L" (" + (target.validated ? L"validated" : L"experimental") + L"), install folder " + dir);
    // the rules table (design section 5): DLSS-NR on AMD, DXVK / vkd3d-proton, two OptiScaler copies, not D3D12, anti-cheat
    if (const Finding* no = target.coexist.refusal())
        Fail(no->tool == Tool::AntiCheat ? AntiCheatMessage(no->evidence) : no->what + L": " + no->message, no->code ? no->code : EXIT_REFUSED);
    // a game that blocks OptiScaler without the player's REFramework (gamerules.h): no install until it is there (an
    // install that is there already is updated or repaired with a warning: DoInstall)
    if (const GameRule* rule = GameRuleFor(target.exe); rule && rule->needsReFramework && !target.installed && !DetectReFramework(target.exeDir).found)
        Fail(NeedsReFrameworkText(*rule), EXIT_REFUSED);
    SystemInfo sys; sys.win = WindowsInfo(); sys.gpu = DetectGpu();
    const std::vector<StoreGame> libraries = r.noScan ? std::vector<StoreGame>{} : ScanLibraries();
    const ModelState model = FindModel(ctx, &target, r.modelDll.empty() ? wstring() : FullPath(r.modelDll), libraries);
    if (model.ok()) LogLine(ctx, L"model source: " + (model.reused ? wstring(L"model data already on this PC") : model.dll + L" (" + model.origin + L")"));
    const Checks c = RunChecks(ctx, target, sys, &model, r.force);
    for (int i = 0; i < L_COUNT; ++i) {
        LogLine(ctx, wstring(c.st[i] == CK_OK ? L"  ok    " : c.st[i] == CK_WARN ? L"  warn  " : c.st[i] == CK_NA ? L"  n/a   " : L"  FAIL  ") + c.text[i]);
        if (c.st[i] == CK_WARN && i != L_CONFLICT) res.warnings.push_back(c.text[i]);
    }
    wstring why;
    if (int code = c.firstFailure(r.force, why)) Fail(why, code);
    if (!target.validated && !r.experimental && !target.installed) Fail(target.name + L" is experimental (not tested end to end). Confirm to install anyway.", EXIT_EXPERIMENTAL);
    // leftover WrongGPU files without a record: cleaned up first only with the player's OK (never automatic)
    std::vector<Leftover> leftovers = target.installed ? std::vector<Leftover>{} : FindLeftovers(ctx, dir);
    // 0.6.1: beside another mod's OptiScaler they are that setup's: never removed first; the install backs up any it replaces
    if (LeftoversUsedByOtherSetup(ctx, dir, leftovers)) {
        res.notes.push_back(L"WrongGPU files " + wstring(kLeftoversOtherSetup) + L" were left alone: " + LeftoverSummary(leftovers) +
                            L" (any the install replaces is backed up and put back when WrongGPU is uninstalled)");
        LogLine(ctx, res.notes.back());
        leftovers.clear();
    }
    if (!leftovers.empty() && !r.removeOld)
        Fail(L"Leftover WrongGPU files without an install record are in this game: " + LeftoverSummary(leftovers) +
             L". Confirm to clean them up first (only files that match WrongGPU's releases are removed), or adopt them as an install.", EXIT_OLD_INSTALL);
    if (target.conflicts.modsFound() && !r.moveMods) {
        wstring list; for (auto& f : target.conflicts.modFiles) list += (list.empty() ? L"" : L", ") + f;
        Fail(L"This game has its own OptiScaler or another upscaler mod (" + list + L"). Switch to WrongGPU's OptiScaler setup? "
             L"Your files are backed up and put back byte for byte when WrongGPU is uninstalled.", EXIT_OTHER_MODS);
    }
    if (target.conflicts.oldFound() && !r.removeOld) Fail(L"An older WrongGPU (DLSS 4.5 AMD) install was found. Confirm to remove it first (its originals are restored byte for byte).", EXIT_OLD_INSTALL);
    if (DWORD e = WriteTest(dir)) Fail(L"Cannot write to the game folder (" + ErrText(e) + L"). Administrator permission is needed.", EXIT_REFUSED);
    // the install in the wrong place (rc6 L2): what goes over to the game (its profile, the OptiScaler tab's journal, the
    // Engine.ini fix and its backup), then it is uninstalled there (the shared model data and the managed list stay)
    Record wrong;
    string wrongFixBackup;
    if (!wrongDir.empty() && !IEq(FullPath(wrongDir), FullPath(dir)) && LoadRecord(RecordPath(wrongDir), wrong) && !wrong.legacy) {
        if (!wrong.exe.empty() && GameRunning(wrong.exe)) Fail(BaseName(wrong.exe) + L" is running. Close it first.", EXIT_REFUSED);
        if (DWORD e = WriteTest(wrongDir)) Fail(L"Cannot write to " + wrongDir + L" (" + ErrText(e) + L"). Administrator permission is needed.", EXIT_REFUSED);
        const Profile carried = GameProfile(wrongDir);
        if (wrong.rrconfig.backup) ReadAll(RrConfigBackupPath(wrongDir), wrongFixBackup);
        ui.log(L"WrongGPU is next to " + BaseName(wrong.exe) + L", not the game's program: taking it out of " + wrongDir + L" first");
        for (auto& w : DoRestore(ctx, wrongDir, ui, false, true)) res.warnings.push_back(w);
        RemoveTree(StateDir(wrongDir));
        RemoveManagedGame(wrongDir);
        if (!IsFile(ProfilePath(dir))) SaveProfile(ProfilePath(dir), carried);  // the game's settings go with it (kept through a failure)
        res.notes.push_back(L"WrongGPU was taken out of " + wrongDir + L" (next to " + BaseName(wrong.exe) + L", not the game's program); its settings went over to " + dir);
        LogLine(ctx, L"launcher: the install next to " + wrong.exe + L" was removed; it goes to " + dir);
    } else wrongDir.clear();
    for (auto& n : CleanLeftovers(ctx, dir, leftovers)) res.notes.push_back(n);
    // what the game had before: its profile (kept through the update) and, for Update, the installed version
    Profile profile;
    bool haveProfile = LoadProfile(ProfilePath(dir), profile);
    if (!haveProfile) {
        string wini, oini;
        if (ReadAll(Join(dir, L"wronggpu.ini"), wini)) { ReadAll(Join(dir, L"OptiScaler.ini"), oini); profile = ImportProfile(wini, oini); haveProfile = true; }
    }
    if (!haveProfile) profile = r.hasProfile ? r.profile : DefaultProfile();
    FoldRrFromGame(dir, profile);  // a Ray Reconstruction mode chosen in the game stays (the install below rewrites its file)
    if (target.installed && !target.record.version.empty() && ctx.data && !IEq(target.record.version, ctx.data->version)) SavePrevious(ctx, dir, target.record, ui);
    std::vector<IniKeyChange> oldJournal = target.installed ? target.record.inikeys : !wrongDir.empty() ? wrong.inikeys : std::vector<IniKeyChange>{};
    // a game's opt-in keys follow its setting (ApplyProfile), not the old file
    oldJournal.erase(std::remove_if(oldJournal.begin(), oldJournal.end(), [&](const IniKeyChange& c) { return OptInKey(target.exe, c); }), oldJournal.end());
    // rc6 L5: values OptiScaler's own menu saved over the launcher's (the journal's keys, JournalOutside) carry over to the new
    // OptiScaler.ini as the player's choices, as the OptiScaler tab's Keep would make them (locked keys stay WrongGPU's)
    if (target.installed && HostedRecord(target.record) && OptiSetupIsOurs(dir, target.record)) {  // (0.6.1: never another setup's values)
        string current;
        if (ReadAll(Join(dir, L"OptiScaler.ini"), current))
            for (auto& o : JournalOutside(current, oldJournal, L"OptiScaler.ini")) {
                if (o.fileValue == "(missing)" || IsOptiLocked(o.section, o.key)) continue;
                for (auto& c : oldJournal)
                    if (IEq(c.file, L"OptiScaler.ini") && c.section == o.section && c.key == o.key) c.ours = o.fileValue;
                LogLine(ctx, L"OptiScaler.ini: [" + FromUtf8(o.section) + L"] " + FromUtf8(o.key) + L" = " + FromUtf8(o.fileValue) + L" (saved by OptiScaler's menu) carried over");
            }
    }
    // the game's Engine.ini fix (rrconfig.h) is not in the game folder: its record line and backup go over to the new record
    const RrConfigFix keepFix = target.installed ? target.record.rrconfig : !wrongDir.empty() ? wrong.rrconfig : RrConfigFix();
    string keepBackup = wrongDir.empty() ? string() : wrongFixBackup;
    if (keepFix.backup && wrongDir.empty()) ReadAll(RrConfigBackupPath(dir), keepBackup);
    Consent consent; consent.moveMods = r.moveMods; consent.removeOld = r.removeOld; consent.movePlugins = r.movePlugins; consent.plugins = r.movePluginsList;
    consent.mods = r.moveModsList;
    try {
        for (auto& w : DoInstall(ctx, target, model, consent, ui, sys.hip)) res.warnings.push_back(w);
    } catch (const SetupError&) {
        if (keepFix.present && !IsFile(RecordPath(dir))) {  // the earlier install went: its Engine.ini change goes with it
            Record gone; gone.exe = target.exe; gone.rrconfig = keepFix;
            std::vector<wstring> n, w;
            RevertRrConfig(ctx, dir, gone, n, w, &keepBackup);
        }
        throw;
    }
    if (keepFix.present) {
        Record m;
        if (LoadRecord(RecordPath(dir), m)) {
            m.rrconfig = keepFix;
            if (keepFix.backup) { EnsureDir(Parent(RrConfigBackupPath(dir))); WriteFileAtomic(RrConfigBackupPath(dir), keepBackup.data(), keepBackup.size()); }
            SaveRecord(RecordPath(dir), m);
        }
    }
    ApplyProfile(ctx, dir, profile, res.notes, &res.warnings);
    if (!oldJournal.empty()) {  // the player's OptiScaler tab choices carried over to the new OptiScaler.ini
        Record m;
        string fresh;
        if (LoadRecord(RecordPath(dir), m) && HostedRecord(m) && ReadAll(Join(dir, L"OptiScaler.ini"), fresh)) {
            const string out = ReplayJournal(fresh, oldJournal, L"OptiScaler.ini", m.inikeys, &res.notes);
            if (out != fresh) {
                WriteFileAtomic(Join(dir, L"OptiScaler.ini"), out.data(), out.size());
                const wstring h = Sha256Bytes(out.data(), out.size());
                for (auto& e : m.e) if (e.action == L"copy" && IEq(e.rel, L"OptiScaler.ini")) e.inst = h;
                res.notes.push_back(L"your OptiScaler settings were carried over");
            }
            SaveRecord(RecordPath(dir), m);
            WriteOwnedFiles(dir, m);
        }
    }
    AddManagedGame(dir);
    const wstring ver = ctx.data ? ctx.data->version : L"", was = target.installed ? target.record.version : L"";
    // a repair puts this release's files back: on an older version it is an update too, and the message says so
    if (!wrongDir.empty()) res.message = L"WrongGPU " + ver + L" moved to the game's program in " + dir + L" (it was next to " + BaseName(wrong.exe) + L" in " + wrongDir + L")";
    else if (r.op == Op::Repair && !was.empty() && !IEq(was, ver)) res.message = L"WrongGPU repaired in " + dir + L" and updated from " + was + L" to " + ver;
    else res.message = L"WrongGPU " + ver + L" " + (r.op == Op::Install && !target.installed ? L"installed" : r.op == Op::Repair ? L"repaired" : L"updated") + L" in " + dir;
    // the earlier install's restore could not put every original back (the ownership rule, record.h): said, with each one
    if (AnyPartly(res.warnings)) res.message = PartlyText(res.message, res.warnings);
}

wstring PartlyText(const wstring& done, const std::vector<wstring>& warnings) {
    size_t n = 0;
    for (auto& w : warnings) if (w.rfind(kPartlyRestored, 0) == 0) ++n;
    return done + L", partly: " + std::to_wstring(n) + L" file(s) could not be put back because another file is at their place now (each is "
           L"listed with where its original is kept)";
}

OpResult RunOp(Context& ctx, const OpRequest& r, Ui& ui) {
    OpResult res;
    try {
        switch (r.op) {
            case Op::Install: case Op::Repair: case Op::Update: InstallOp(ctx, r, ui, res); break;
            case Op::Uninstall: {
                Record m;
                if (r.dir.empty() || !LoadRecord(RecordPath(r.dir), m) || m.legacy) Fail(L"No WrongGPU installation record was found in " + r.dir + L".", EXIT_NOT_INSTALLED);
                if (!m.exe.empty() && GameRunning(m.exe)) Fail(BaseName(m.exe) + L" is running. Close it first.", EXIT_REFUSED);
                for (auto& w : DoRestore(ctx, r.dir, ui, false, false)) res.warnings.push_back(w);
                RemoveManagedGame(r.dir);
                // the ownership rule (record.h): "back" only when every file was put back; else "partly", each one in the warnings
                res.message = AnyPartly(res.warnings) ? PartlyText(L"WrongGPU was uninstalled from " + r.dir, res.warnings)
                                                      : L"WrongGPU was uninstalled from " + r.dir + L"; its original files are back";
                break;
            }
            case Op::Detach: {  // the launcher is uninstalled, the game keeps WrongGPU: its own Apps & features entry again
                Record m;
                if (r.dir.empty() || !LoadRecord(RecordPath(r.dir), m) || m.legacy) Fail(L"No WrongGPU installation record was found in " + r.dir + L".", EXIT_NOT_INSTALLED);
                if (m.regkey.empty()) { m.regkey = RegKeyFor(r.dir); SaveRecord(RecordPath(r.dir), m); }
                RegisterUninstall(ctx, m, StateDir(r.dir), GameDisplayName(m.exe));
                res.message = L"WrongGPU stays in " + r.dir + L" with its own uninstaller";
                break;
            }
            case Op::Rollback: Rollback(ctx, r.dir, ui, res); break;
            case Op::Apply: {
                if (r.rrConfigOnly) {  // the game's Engine.ini alone (UI review r3 #1)
                    ApplyRrConfigOnly(ctx, r.dir, res.notes, &res.warnings, r.rrConfigSet);
                    res.message = L"Engine.ini setting applied; it takes effect the next time the game starts";
                    break;
                }
                if (!r.optiOnly) ApplyProfile(ctx, r.dir, r.hasProfile ? r.profile : GameProfile(r.dir), res.notes, &res.warnings);
                ApplyOptiTab(ctx, r.dir, r, res.notes);
                res.message = L"Settings applied; they take effect the next time the game starts";
                break;
            }
            case Op::CleanLeftovers: case Op::AdoptLeftovers: {
                const std::vector<Leftover> lo = FindLeftovers(ctx, r.dir);
                if (lo.empty()) Fail(L"There are no leftover WrongGPU files (without an install record) in " + r.dir + L".", EXIT_NOT_INSTALLED);
                if (!r.exe.empty() && GameRunning(r.exe)) Fail(BaseName(r.exe) + L" is running. Close it first.", EXIT_REFUSED);
                if (!r.otherSetupOk && LeftoversUsedByOtherSetup(ctx, r.dir, lo))  // 0.6.1: left alone unless the player confirms (--confirm-other-setup)
                    Fail(L"These WrongGPU files are " + wstring(kLeftoversOtherSetup) + L", so they are left alone: removing or adopting them can break that mod. "
                         L"Confirm to " + (r.op == Op::CleanLeftovers ? L"remove" : L"adopt") + L" them anyway.", EXIT_REFUSED);
                if (r.op == Op::CleanLeftovers) {
                    res.notes = CleanLeftovers(ctx, r.dir, lo);
                    res.message = L"Leftover WrongGPU files removed from " + r.dir + L" (" + LeftoverSummary(lo) + L")";
                } else {
                    res.notes = AdoptLeftovers(ctx, r.exe, r.dir, lo);
                    res.message = L"Adopted as a WrongGPU install: " + LeftoverSummary(lo);
                }
                break;
            }
            case Op::Adopt: {
                bool repair = false;
                res.notes = AdoptInstall(ctx, r.dir, &repair);
                res.message = repair ? L"Adopted; repair suggested" : L"Adopted";
                break;
            }
        }
        LogLine(ctx, wstring(OpName(r.op)) + L": " + res.message);
    } catch (const SetupError& e) {
        res.code = e.code ? e.code : EXIT_FAILED;
        res.message = e.msg;
        LogLine(ctx, wstring(OpName(r.op)) + L" ERROR: " + e.msg);
    }
    return res;
}

// ---- job file ------------------------------------------------------------------------------------------------------
string JobText(const OpRequest& r, const wstring& localAppData, const wstring& logFile, const wstring& userSid, int perGameEntry, const wstring& documents) {
    string t = "; WrongGPU launcher job (run by: WrongGPU.exe apply --job <this file> --result <file>)\r\n";
    if (!userSid.empty()) t += "usersid=" + ToUtf8(userSid) + "\r\n";
    if (perGameEntry >= 0) t += string("perentry=") + (perGameEntry ? "1" : "0") + "\r\n";
    t += "op=" + ToUtf8(OpName(r.op)) + "\r\nexe=" + ToUtf8(r.exe) + "\r\ndir=" + ToUtf8(r.dir) + "\r\nopti=" + ToUtf8(r.optiPackage) +
         "\r\nmodel=" + ToUtf8(r.modelDll) + "\r\nlocalappdata=" + ToUtf8(localAppData) + "\r\nlog=" + ToUtf8(logFile) + "\r\n";
    if (!documents.empty()) t += "documents=" + ToUtf8(documents) + "\r\n";
    t += string("movemods=") + (r.moveMods ? "1" : "0") + "\r\nremoveold=" + (r.removeOld ? "1" : "0") + "\r\nexperimental=" + (r.experimental ? "1" : "0") +
         "\r\nforce=" + (r.force ? "1" : "0") + "\r\nnoscan=" + (r.noScan ? "1" : "0") + "\r\n";
    if (r.hasProfile)
        for (size_t i = 0; i < kSettingCount; ++i) t += string("profile.") + kSettings[i].key + "=" + r.profile.get(kSettings[i].key) + "\r\n";
    for (auto& e : r.optiEdits) t += "optikey." + e.first + "=" + e.second + "\r\n";
    if (r.movePlugins) t += "moveplugins=1\r\n";
    // the files the player agreed to move, each with the SHA-256 it had then: "<path>|<sha256>"
    for (size_t i = 0; i < r.movePluginsList.size(); ++i)
        t += "moveplugin." + std::to_string(i) + "=" + ToUtf8(r.movePluginsList[i].first) + "|" + ToUtf8(r.movePluginsList[i].second) + "\r\n";
    for (size_t i = 0; i < r.moveModsList.size(); ++i)
        t += "movemod." + std::to_string(i) + "=" + ToUtf8(r.moveModsList[i].first) + "|" + ToUtf8(r.moveModsList[i].second) + "\r\n";
    if (r.otherSetupOk) t += "othersetupok=1\r\n";
    if (r.optiReset) t += "optireset=1\r\n";
    if (r.optiKeep) t += "optikeep=1\r\n";
    if (r.optiOnly) t += "optionly=1\r\n";
    if (r.rrConfigOnly) t += "rrconfigonly=1\r\n";
    if (!r.rrConfigSet.empty()) t += "rrconfigset=" + r.rrConfigSet + "\r\n";
    return t;
}
bool ParseJob(const string& text, OpRequest& r, wstring& localAppData, wstring& logFile, wstring* why, wstring* userSid, int* perGameEntry, wstring* documents) {
    wstring docs;
    bool haveOp = false;
    r = OpRequest();
    if (userSid) userSid->clear();
    if (perGameEntry) *perGameEntry = -1;
    for (auto& kv : KeyValues(text)) {
        const string& k = kv.first; const wstring v = FromUtf8(kv.second);
        if (k == "op") { if (!ParseOp(v, r.op)) { if (why) *why = L"unknown operation " + v; return false; } haveOp = true; }
        else if (k == "exe") r.exe = v;
        else if (k == "dir") r.dir = v;
        else if (k == "opti") r.optiPackage = v;
        else if (k == "model") r.modelDll = v;
        else if (k == "localappdata") localAppData = v;
        else if (k == "documents") docs = v;
        else if (k == "log") logFile = v;
        else if (k == "usersid") { if (userSid) *userSid = v; }
        else if (k == "perentry") { if (perGameEntry) *perGameEntry = Truthy(kv.second) ? 1 : 0; }
        else if (k == "movemods") r.moveMods = Truthy(kv.second);
        else if (k == "removeold") r.removeOld = Truthy(kv.second);
        else if (k == "experimental") r.experimental = Truthy(kv.second);
        else if (k == "force") r.force = Truthy(kv.second);
        else if (k == "noscan") r.noScan = Truthy(kv.second);
        else if (k == "moveplugins") r.movePlugins = Truthy(kv.second);
        else if (k.rfind("moveplugin.", 0) == 0 || k.rfind("movemod.", 0) == 0) {  // "<path>|<sha256>" (no hash: agreed to nothing)
            const size_t bar = v.rfind(L'|');
            (k[4] == 'p' ? r.movePluginsList : r.moveModsList).push_back({bar == wstring::npos ? v : v.substr(0, bar), bar == wstring::npos ? wstring() : v.substr(bar + 1)});
        }
        else if (k == "othersetupok") r.otherSetupOk = Truthy(kv.second);
        else if (k == "optireset") r.optiReset = Truthy(kv.second);
        else if (k == "optikeep") r.optiKeep = Truthy(kv.second);
        else if (k == "optionly") r.optiOnly = Truthy(kv.second);
        else if (k == "rrconfigonly") r.rrConfigOnly = Truthy(kv.second);
        else if (k == "rrconfigset") r.rrConfigSet = kv.second;
        else if (k.rfind("optikey.", 0) == 0) r.optiEdits.push_back({k.substr(8), kv.second});
        else if (k.rfind("profile.", 0) == 0) {
            if (!r.hasProfile) { r.profile = DefaultProfile(); r.hasProfile = true; }
            if (!SetSetting(r.profile, k.substr(8), kv.second, why)) return false;
        }
    }
    if (!haveOp) { if (why) *why = L"the job names no operation"; return false; }
    // explicit absolute paths only: the worker never guesses from its own (possibly another account's) environment
    for (const wstring* p : {&r.exe, &r.dir, &localAppData, &docs})
        if (!p->empty() && !(p->size() > 2 && (*p)[1] == L':' && ((*p)[2] == L'\\' || (*p)[2] == L'/')) && p->rfind(L"\\\\", 0) != 0) {
            if (why) *why = L"not an absolute path: " + *p;
            return false;
        }
    if (localAppData.empty()) { if (why) *why = L"the job names no localappdata folder"; return false; }
    if (documents) *documents = docs;
    return true;
}
string BatchText(const std::vector<std::pair<wstring, wstring>>& jobs, bool stop) {
    string t = "; WrongGPU launcher batch (run by: WrongGPU.exe apply --batch <this file>)\r\n";
    if (stop) t += "stop=1\r\n";
    for (auto& j : jobs) t += "job=" + ToUtf8(j.first) + "\r\nresult=" + ToUtf8(j.second) + "\r\n";
    return t;
}
bool ParseBatch(const string& text, std::vector<std::pair<wstring, wstring>>& jobs, bool& stop) {
    jobs.clear();
    stop = false;
    wstring job;
    for (auto& kv : KeyValues(text)) {
        if (kv.first == "stop") stop = Truthy(kv.second);
        else if (kv.first == "job") job = FromUtf8(kv.second);
        else if (kv.first == "result" && !job.empty()) { jobs.push_back({job, FromUtf8(kv.second)}); job.clear(); }
    }
    return !jobs.empty();
}
string ResultText(const OpResult& res) {
    string t = "code=" + std::to_string(res.code) + "\r\nmessage=" + ToUtf8(res.message) + "\r\n";
    for (auto& w : res.warnings) t += "warning=" + ToUtf8(w) + "\r\n";
    for (auto& n : res.notes) t += "note=" + ToUtf8(n) + "\r\n";
    return t;
}
bool ParseResult(const string& text, OpResult& res) {
    bool any = false;
    res = OpResult();
    for (auto& kv : KeyValues(text)) {
        if (kv.first == "code") { res.code = atoi(kv.second.c_str()); any = true; }
        else if (kv.first == "message") res.message = FromUtf8(kv.second);
        else if (kv.first == "warning") res.warnings.push_back(FromUtf8(kv.second));
        else if (kv.first == "note") res.notes.push_back(FromUtf8(kv.second));
    }
    return any;
}

// ---- the launcher's own install ------------------------------------------------------------------------------------
static wstring StartMenuDir(const wstring& startMenu) {
    if (startMenu == L"-") return L"";
    if (!startMenu.empty()) return startMenu;
#ifdef WRONGGPU_TESTING
    // test builds (their Apps & features entries go to the test registry root) never touch the player's Start menu: a
    // sandbox's "Install for this user" would otherwise point the real WrongGPU shortcut at the sandbox's copy
    wstring base = EnvVar(L"LOCALAPPDATA");
    if (base.empty()) { wchar_t b[MAX_PATH]; GetTempPathW(MAX_PATH, b); base = b; }
    return Join(base, L"WrongGPU-TestStartMenu");
#else
    PWSTR p = nullptr;
    wstring r;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Programs, 0, nullptr, &p))) r = p;
    CoTaskMemFree(p);
    return r;
#endif
}
static bool MakeShortcut(const wstring& lnk, const wstring& target, wstring* why) {
    // COM on this thread for the call (the window runs Install for this user on a worker thread of its own); a thread that
    // has it already keeps its apartment (RPC_E_CHANGED_MODE: usable as it is, nothing to undo)
    const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    struct CoEnd { HRESULT hr; ~CoEnd() { if (SUCCEEDED(hr)) CoUninitialize(); } } coEnd{co};
    IShellLinkW* sl = nullptr;
    if (FAILED(CoCreateInstance(CLSID_ShellLink, nullptr, CLSCTX_INPROC_SERVER, IID_IShellLinkW, (void**)&sl))) {
        if (why) *why = L"cannot write the shortcut " + lnk + L": the shell's link object is not available";
        return false;
    }
    sl->SetPath(target.c_str());
    sl->SetWorkingDirectory(Parent(target).c_str());
    sl->SetDescription(L"WrongGPU: Model M upscaling for your games");
    sl->SetIconLocation(target.c_str(), 0);
    IPersistFile* pf = nullptr;
    bool ok = SUCCEEDED(sl->QueryInterface(IID_IPersistFile, (void**)&pf)) && SUCCEEDED(pf->Save(lnk.c_str(), TRUE));
    if (pf) pf->Release();
    sl->Release();
    if (!ok && why) *why = L"cannot write the shortcut " + lnk;
    return ok;
}
// Deletes `path` (a file, or a folder and what is in it) once nothing holds it any more: a program that still runs can be
// renamed but not deleted, so a hidden cmd.exe tries once a second, for up to an hour, then removes `thenDir` if it is
// empty. Nothing is shown and nothing waits for it. Each one is remembered: RetryCleanupsAtExit hands what is still there
// to a fresh cleanup as this program ends (its window may stay open for longer than the hour, its result note unread).
// `upTo`: thenDir's parents up to that folder go too while they are empty (the %LOCALAPPDATA%\Programs a first install made).
// `atSignIn`: noted in the user's RunOnce as well, so a session that ends first (the uninstall's note still open at sign-out,
// both processes ended) finishes it at the next sign-in: no install is left to clean it up later. The hidden cleanup takes that
// note away once it is done.
struct PendingCleanup { wstring path, thenDir, upTo; bool tree, atSignIn; };
static std::mutex cleanupsLock;
static std::vector<PendingCleanup> cleanups;
static void RemoveEmptyUpTo(const wstring& dir, const wstring& top);
// The user's RunOnce key (test builds: one beside their own Apps & features root, never the real one)
static wstring RunOnceKey() {
#ifdef WRONGGPU_TESTING
    const wstring k = UninstallRoot();  // ...\<hash>\Uninstall\ -> ...\<hash>\RunOnce
    const size_t cut = k.size() > 2 ? k.rfind(L'\\', k.size() - 2) : wstring::npos;
    return cut == wstring::npos ? wstring() : k.substr(0, cut + 1) + L"RunOnce";
#else
    return L"Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce";
#endif
}
static void DeleteWhenFree(Context& ctx, const wstring& path, bool tree, const wstring& thenDir, const wstring& upTo = L"", bool atSignIn = false) {
    {
        std::lock_guard<std::mutex> lock(cleanupsLock);
        cleanups.push_back({path, thenDir, upTo, tree, atSignIn});
    }
    wchar_t sys[MAX_PATH]; GetSystemDirectoryW(sys, MAX_PATH);
    const wstring cmdExe = Join(sys, L"cmd.exe"), pingExe = Join(sys, L"PING.EXE");
    const wstring q = L"\"" + path + L"\"";
    // the folders that go after it, while empty (rmdir without /s): thenDir, then its parents up to upTo
    wstring rmdirs;
    if (!thenDir.empty()) {
        rmdirs = L"rmdir \"" + thenDir + L"\" 2>nul & ";
        if (!upTo.empty() && StartsWithI(FullPath(thenDir), FullPath(upTo) + L"\\"))
            for (wstring a = Parent(thenDir); !a.empty() && StartsWithI(FullPath(a), FullPath(upTo)); a = Parent(a)) {
                rmdirs += L"rmdir \"" + a + L"\" 2>nul & ";
                if (IEq(FullPath(a), FullPath(upTo))) break;
            }
    }
    const wstring noteName = L"WrongGPU cleanup " + BaseName(path), noteKey = RunOnceKey();
    wstring unnote;
    if (atSignIn && !noteKey.empty()) {
        const wstring note = L"\"" + cmdExe + L"\" /D /C \"" + (tree ? L"rmdir /s /q " : L"del /f /q ") + q + L" 2>nul & " + rmdirs + L"exit\"";
        HKEY k;
        // RunOnce runs a command line of at most 260 characters: a longer one is left out (the cleanup below still runs)
        if (note.size() < 260 && RegCreateKeyExW(HKEY_CURRENT_USER, noteKey.c_str(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr) == ERROR_SUCCESS) {
            if (RegSetValueExW(k, noteName.c_str(), 0, REG_SZ, (const BYTE*)note.c_str(), (DWORD)((note.size() + 1) * 2)) == ERROR_SUCCESS)
                unnote = L"\"" + Join(sys, L"reg.exe") + L"\" delete \"HKCU\\" + noteKey + L"\" /v \"" + noteName + L"\" /f >nul 2>nul & ";
            RegCloseKey(k);
        }
    }
    const wstring gone = rmdirs.empty() && unnote.empty() ? wstring(L"exit") : L"(" + rmdirs + unnote + L"exit)";
    const wstring steps = L"for /l %i in (1,1,3600) do @(if not exist " + q + L" " + gone + L") & @(" + (tree ? L"rmdir /s /q " : L"del /f /q ") + q +
                          L" 2>nul) & @(if exist " + q + L" \"" + pingExe + L"\" -n 2 127.0.0.1 >nul)";
    wstring cmd = L"\"" + cmdExe + L"\" /D /C \"" + steps + L"\"";
    std::vector<wchar_t> buf(cmd.begin(), cmd.end()); buf.push_back(0);
    STARTUPINFOW si{}; si.cb = sizeof(si); si.dwFlags = STARTF_USESHOWWINDOW; si.wShowWindow = SW_HIDE;
    PROCESS_INFORMATION pi{};
    if (CreateProcessW(cmdExe.c_str(), buf.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, sys, &si, &pi)) {
        CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
        LogLine(ctx, L"removed once it is free: " + path);
    } else {
        LogLine(ctx, L"could not hand " + path + L" to the hidden cleanup (error " + std::to_wstring(GetLastError()) + L")");
    }
}
void RetryCleanupsAtExit(Context& ctx) {
    std::vector<PendingCleanup> left;
    {
        std::lock_guard<std::mutex> lock(cleanupsLock);
        left.swap(cleanups);
    }
    for (auto& c : left)
        if (Exists(c.path)) DeleteWhenFree(ctx, c.path, c.tree, c.thenDir, c.upTo, c.atSignIn);
        else if (!c.thenDir.empty()) RemoveEmptyUpTo(c.thenDir, c.upTo);  // gone already: only its folders, if empty
}
static wstring LauncherEntryKey() { return UninstallRoot() + kLauncherEntry; }
// The launcher's one Apps & features entry. Its Uninstall opens the launcher's own uninstall (the window asks whether the
// games keep WrongGPU and shows how it went); the size counts the program and the model data it manages.
static bool WriteLauncherEntry(Context& ctx, const wstring& exe, const wstring& shortcut) {
    HKEY k;
    if (RegCreateKeyExW(UninstallHive(), LauncherEntryKey().c_str(), 0, nullptr, 0, KEY_WRITE | KEY_READ, nullptr, &k, nullptr) != ERROR_SUCCESS) return false;
    auto setS = [&](const wchar_t* n, const wstring& v) { RegSetValueExW(k, n, 0, REG_SZ, (const BYTE*)v.c_str(), (DWORD)((v.size() + 1) * 2)); };
    auto setD = [&](const wchar_t* n, DWORD v) { RegSetValueExW(k, n, 0, REG_DWORD, (const BYTE*)&v, 4); };
    setS(L"DisplayName", kAppName);
    setS(L"DisplayVersion", ctx.data ? ctx.data->version : L"");
    setS(L"Publisher", L"WrongGPU");
    setS(L"InstallLocation", Parent(exe));
    setS(L"UninstallString", L"\"" + exe + L"\" uninstall-launcher");
    setS(L"DisplayIcon", exe);
    setS(L"StartMenuShortcut", shortcut);
    if (RegString(UninstallHive(), LauncherEntryKey().c_str(), L"InstallDate").empty()) {  // the first install's date (a replace keeps it)
        SYSTEMTIME t; GetLocalTime(&t);
        wchar_t d[16]; swprintf(d, 16, L"%04d%02d%02d", t.wYear, t.wMonth, t.wDay);
        setS(L"InstallDate", d);
    }
    setD(L"NoModify", 1); setD(L"NoRepair", 1);
    // the program and the data its uninstall removes with it: Model M's and the Ray Reconstruction data
    unsigned long long size = FileSize(exe);
    for (const wstring& f : {ModelFile(ctx), RrDataFile()}) if (IsFile(f)) size += FileSize(f);
    setD(L"EstimatedSize", (DWORD)(size / 1024));
    RegCloseKey(k);
    return true;
}
void RefreshLauncherEntry(Context& ctx, const wstring& selfExe) {
    wstring exe;
    if (!LauncherInstalled(&exe) || !IEq(FullPath(exe), FullPath(selfExe))) return;
    WriteLauncherEntry(ctx, exe, RegString(UninstallHive(), LauncherEntryKey().c_str(), L"StartMenuShortcut"));
    DeleteFileW((exe + L".old").c_str());  // the copy an earlier replace set aside while it ran
}
// The entry's note of the topmost folder above the install folder that the first install made (none: they were there)
static const wchar_t* const kMadeFolder = L"WrongGPUMadeFolder";
// `dir` and its parents up to `top` (empty: dir alone) removed while they are empty: never a folder that holds anything
static void RemoveEmptyUpTo(const wstring& dir, const wstring& top) {
    if (!RemoveDirectoryW(dir.c_str()) && IsDir(dir)) return;
    if (top.empty()) return;
    for (wstring a = Parent(dir); !a.empty() && a.size() >= top.size() && StartsWithI(FullPath(a), FullPath(top)); a = Parent(a)) {
        if (!RemoveDirectoryW(a.c_str())) return;
        if (IEq(FullPath(a), FullPath(top))) return;
    }
}
bool SelfInstall(Context& ctx, const wstring& selfExe, const wstring& dirIn, const wstring& startMenu, wstring* why) {
    const wstring dir = dirIn.empty() ? DefaultInstallDir() : FullPath(dirIn);
    const wstring exe = Join(dir, L"WrongGPU.exe");
    // a first install that stops after its program is in place (the shortcut or the entry cannot be written) is undone: no
    // WrongGPU.exe is left that no entry uninstalls. A replace keeps the new program (the old one is gone) and its entry.
    HKEY had = nullptr;
    const bool entryBefore = RegOpenKeyExW(UninstallHive(), LauncherEntryKey().c_str(), 0, KEY_READ, &had) == ERROR_SUCCESS;
    if (had) RegCloseKey(had);
    const bool dirBefore = IsDir(dir), exeBefore = IsFile(exe);
    // the folders above it this install makes (%LOCALAPPDATA%\Programs on a fresh profile): the topmost of them, noted in the
    // entry, so the uninstall (and the undo of a first install) leaves no empty folder of WrongGPU's behind
    wstring madeAbove;
    for (wstring a = Parent(dir); !a.empty() && a.size() > 3 && !IsDir(a); a = Parent(a)) madeAbove = a;
    wstring lnkMade;  // the Start-menu shortcut this install wrote (a first install that is undone removes it too)
    auto undoFirst = [&](const wstring& what) {
        if (entryBefore || exeBefore || IEq(FullPath(selfExe), FullPath(exe))) return what;
        RegDeleteKeyW(UninstallHive(), LauncherEntryKey().c_str());
        if (!lnkMade.empty()) DeleteFileW(lnkMade.c_str());
        DeleteFileW(exe.c_str());
        if (!dirBefore) RemoveEmptyUpTo(dir, madeAbove);
        return what + L"; nothing was installed";
    };
    try {
        EnsureDir(dir);
        if (!IEq(FullPath(selfExe), FullPath(exe))) {
            // copied next to it and checked first, then moved into place: a copy that stops half way never leaves a broken
            // WrongGPU.exe behind the Start menu entry. An installed copy that still runs (its window open) is renamed aside,
            // as Windows allows for a running program, and deleted once it closes.
            const wstring fresh = exe + L".new", old = exe + L".old";
            CopyChecked(selfExe, fresh);
            if (Sha256File(fresh) != Sha256File(selfExe)) { DeleteFileW(fresh.c_str()); Fail(L"verification failed after copying the launcher"); }
            // the copy of a downloaded file carries its Mark of the Web (CopyFileW copies the streams too): the installed
            // program is a clean copy, so starting it from the Start menu is not a fresh download's start each time
            DeleteFileW((fresh + L":Zone.Identifier").c_str());
            DeleteFileW(old.c_str());
            bool aside = false;
            if (IsFile(exe)) {
                SetFileAttributesW(exe.c_str(), FILE_ATTRIBUTE_NORMAL);
                if (!MoveFileExW(exe.c_str(), old.c_str(), MOVEFILE_REPLACE_EXISTING)) {
                    const DWORD e = GetLastError();
                    DeleteFileW(fresh.c_str());
                    Fail(L"cannot set the installed launcher aside: " + ErrText(e));
                }
                aside = true;
            }
            if (!MoveFileExW(fresh.c_str(), exe.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                const DWORD e = GetLastError();
                if (aside) MoveFileExW(old.c_str(), exe.c_str(), MOVEFILE_REPLACE_EXISTING);
                DeleteFileW(fresh.c_str());
                Fail(L"cannot put the new launcher in place: " + ErrText(e));
            }
            if (aside && !DeleteFileW(old.c_str())) DeleteWhenFree(ctx, old, false, L"");
        }
        // no folder named (a replace by a newer download, Install for this user): the shortcut stays where the installed
        // copy's entry has it (a self-install --startmenu=<folder>, or none with "-"), so its uninstall still finds it
        wstring menuArg = startMenu;
        HKEY ek = nullptr;
        if (menuArg.empty() && RegOpenKeyExW(UninstallHive(), LauncherEntryKey().c_str(), 0, KEY_READ, &ek) == ERROR_SUCCESS) {
            DWORD type = 0;
            if (RegQueryValueExW(ek, L"StartMenuShortcut", nullptr, &type, nullptr, nullptr) == ERROR_SUCCESS && type == REG_SZ) {
                const wstring lnk = RegString(UninstallHive(), LauncherEntryKey().c_str(), L"StartMenuShortcut");
                menuArg = lnk.empty() ? wstring(L"-") : Parent(lnk);
            }
            RegCloseKey(ek);
        }
        const wstring menu = StartMenuDir(menuArg);
        wstring lnkWhy;
        bool lnkOk = true;
        if (!menu.empty()) {
            try { EnsureDir(menu); } catch (const SetupError& e) { lnkWhy = L"cannot write the shortcut: " + e.msg; lnkOk = false; }
            if (lnkOk) lnkOk = MakeShortcut(Join(menu, L"WrongGPU.lnk"), exe, &lnkWhy);
            if (lnkOk) lnkMade = Join(menu, L"WrongGPU.lnk");
        }
        if (!lnkOk && !entryBefore && !exeBefore) { if (why) *why = undoFirst(lnkWhy); return false; }
        // a replace whose shortcut failed still gets its entry (the new version, its uninstall), then says so
        if (!WriteLauncherEntry(ctx, exe, menu.empty() ? wstring() : Join(menu, L"WrongGPU.lnk"))) {
            if (why) *why = undoFirst(L"cannot create the Apps & features entry");
            if (!lnkOk && why) *why = lnkWhy + L"; " + *why;
            return false;
        }
        if (!madeAbove.empty()) {  // kept by a later replace (it only adds values), read by the uninstall
            HKEY mk = nullptr;
            if (RegOpenKeyExW(UninstallHive(), LauncherEntryKey().c_str(), 0, KEY_WRITE, &mk) == ERROR_SUCCESS) {
                RegSetValueExW(mk, kMadeFolder, 0, REG_SZ, (const BYTE*)madeAbove.c_str(), (DWORD)((madeAbove.size() + 1) * 2));
                RegCloseKey(mk);
            }
        }
        if (!lnkOk) { if (why) *why = lnkWhy; LogLine(ctx, L"launcher replaced, but " + lnkWhy); return false; }
        LogLine(ctx, L"launcher installed for this user: " + exe + (menu.empty() ? wstring() : L" (Start menu: " + menu + L")"));
        return true;
    } catch (const SetupError& e) {  // before the program was in place: nothing of it is there (the folders it made go, if empty)
        if (why) *why = e.msg;
        if (!dirBefore) RemoveEmptyUpTo(dir, madeAbove);
        return false;
    }
}

// The launcher's own files in its data folder, by their names only (a --data folder may hold other things).
static void RemoveLauncherFiles(const wstring& dir) {
    for (auto n : {L"launcher.ini", L"launcher.log", L"library-list.txt"}) DeleteFileW(Join(dir, n).c_str());  // library-list.txt: the window's last list
    const std::pair<const wchar_t*, std::vector<const wchar_t*>> sub[] = {{L"logs", {L"last-operation.log", L"elevated-*.log"}}, {L"jobs", {L"job-*.txt", L"result-*.txt", L"batch-*.txt"}}};
    for (auto& s : sub) {
        const wstring d = Join(dir, s.first);
        for (auto pat : s.second) for (auto& f : ListFiles(d, pat)) DeleteFileW(f.c_str());
        RemoveDirectoryW(d.c_str());
    }
}

bool AnyGameHasWrongGPU() { return !OtherGameWithWrongGPU(L"").empty(); }
LauncherUninstallLook LookBeforeLauncherUninstall(Context& ctx) {
    LauncherUninstallLook l;
    std::vector<wstring> locs = RegisteredInstalls();
    for (auto& d : ManagedGames()) locs.push_back(d);
    for (auto& loc : locs) {  // as OtherGameWithWrongGPU counts them (a game on a drive that is not connected counts)
        if (loc.empty()) continue;
        if (IsFile(RecordPath(loc))) l.anyGame = true;
        else if (DriveUnreachable(loc)) l.anyGame = l.offlineGame = true;
    }
    l.portableUses = PortableLauncherUsesSharedData();
    l.modelData = IsFile(ModelFile(ctx));
    l.rrData = IsFile(RrDataFile());
    return l;
}
SharedDataLook LookAtSharedData(Context& ctx) {
    SharedDataLook s;
    const wstring shared = AppDataPath();
    s.there = IsDir(shared);
    if (!s.there) return s;
    const LauncherUninstallLook l = LookBeforeLauncherUninstall(ctx);
    s.anyGame = l.anyGame; s.modelData = l.modelData; s.rrData = l.rrData;
    s.installed = LauncherInstalled() || IsFile(Join(shared, L"launcher.ini"));
    return s;
}
// Log a linked root before RemoveTree unlinks it; never queue recursive cleanup for a remaining link.
static bool RemoveSharedTree(Context& ctx, const wstring& dir) {
    const DWORD attrs = GetFileAttributesW(dir.c_str());
    const bool link = attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_REPARSE_POINT);
    if (link) LogLine(ctx, L"cleanup: removing a data folder link; its target will be kept: " + dir);
    RemoveTree(dir);
    return !link;
}

bool RemoveSharedData(Context& ctx, wstring* why) {
    try {
        const SharedDataLook s = LookAtSharedData(ctx);
        if (!s.there) { LogLine(ctx, L"remove data: no shared data folder: nothing to remove"); return true; }
        if (s.installed) { if (why) *why = L"WrongGPU is installed for this user: its own uninstall removes this data. Nothing was removed."; return false; }
        if (s.anyGame) { if (why) *why = L"a game still has WrongGPU and uses this data. Uninstall WrongGPU from it first. Nothing was removed."; return false; }
        const wstring shared = AppDataPath();
        const DWORD sharedAttrs = GetFileAttributesW(shared.c_str());
        if (sharedAttrs != INVALID_FILE_ATTRIBUTES && (sharedAttrs & FILE_ATTRIBUTE_REPARSE_POINT)) {
            if (!RemoveDirectoryW(shared.c_str())) {
                if (why) *why = L"Could not remove the WrongGPU data folder link: " + ErrText(GetLastError());
                return false;
            }
            LogLine(ctx, L"remove data: removed the WrongGPU data folder link; its target was kept: " + shared);
            return true;
        }
        const wstring trash = shared + L".removed-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64());
        LogLine(ctx, L"remove data: " + shared + L" (no game and no installed launcher use it)");
        {   // nothing is written into the folder that goes from here on (a log line would bring it back)
            std::lock_guard<std::mutex> lock(ctx.logMutex);
            if (StartsWithI(ctx.logFile, shared + L"\\")) ctx.logFile.clear();
        }
        // a file held open for a moment (the window's health check hashing the model) lets go soon: tried for two seconds
        DWORD err = 0;
        bool moved = false;
        for (int i = 0; i < 10 && !moved; ++i) {
            if (MoveFileExW(shared.c_str(), trash.c_str(), MOVEFILE_WRITE_THROUGH)) moved = true;
            else { err = GetLastError(); Sleep(200); }
        }
        if (!moved) {
            if (why) *why = L"a file in WrongGPU's data folder is in use (" + ErrText(err) + L"). Close what uses it and try again. Nothing was removed.";
            LogLine(ctx, L"remove data: " + shared + L" could not be moved aside: " + ErrText(err));
            return false;
        }
        const bool ordinaryTrash = RemoveSharedTree(ctx, trash);
        if (!ordinaryTrash && IsDir(trash)) { if (why) *why = L"Could not remove the WrongGPU data folder link."; return false; }
        if (ordinaryTrash && IsDir(trash)) DeleteWhenFree(ctx, trash, true, L"");  // what is still held goes once it is free
        LogLine(ctx, L"remove data: removed " + shared);
        return true;
    } catch (const SetupError& e) {
        if (why) *why = e.msg;
        return false;
    } catch (...) {
        if (why) *why = L"WrongGPU's data could not be removed. Nothing was removed.";
        return false;
    }
}

// A managed game whose drive is not connected: its own Apps & features entry (as Detach gives it) from its folder alone, its
// name from the folder (the game's files cannot be read); none when it has one already
static void RegisterUnreachableGame(Context& ctx, const wstring& dir) {
    Record m;
    m.game = dir;
    m.regkey = RegKeyFor(dir);
    HKEY had = nullptr;
    if (RegOpenKeyExW(UninstallHive(), (UninstallRoot() + m.regkey).c_str(), 0, KEY_READ, &had) == ERROR_SUCCESS) { RegCloseKey(had); return; }
    const wstring root = GameRoot(dir);
    try {
        RegisterUninstall(ctx, m, StateDir(dir), BaseName(root).empty() ? dir : BaseName(root));
        LogLine(ctx, L"launcher uninstall: " + dir + L" keeps WrongGPU with its own Apps & features entry (its drive is not connected)");
    } catch (const SetupError& e) { LogLine(ctx, L"launcher uninstall: no entry of its own for " + dir + L": " + e.msg); }
}
bool SelfUninstall(Context& ctx, const LauncherPaths& p, bool removeGames, const wstring& startMenu, Ui& ui, wstring* why, const GameOps& gameOps,
                   std::vector<wstring>* unreachable) {
    try {
        wstring installedExe;
        const bool installed = LauncherInstalled(&installedExe);
        const wstring key = LauncherEntryKey();
        const wstring lnk = RegString(UninstallHive(), key.c_str(), L"StartMenuShortcut");
        auto nameOf = [](const wstring& dir, const Record& m) { return m.exe.empty() ? dir : GameDisplayName(m.exe); };
        // 1. The games, before anything of the launcher changes. A game that cannot be done stops it there: the launcher
        //    stays installed with all its settings, the games done so far stay done, and trying again carries on.
        std::vector<std::pair<wstring, Record>> games;
        std::vector<wstring> offline;  // on a drive that is not connected now: left as it is (neither restored nor detached)
        const DWORD sharedStartAttrs = GetFileAttributesW(AppDataPath().c_str());
        const bool sharedStartsAsLink = sharedStartAttrs != INVALID_FILE_ATTRIBUTES && (sharedStartAttrs & FILE_ATTRIBUTE_REPARSE_POINT);
        // Do not rewrite a managed-games list reached through a linked shared-data root.
        for (auto& dir : sharedStartsAsLink ? std::vector<wstring>{} : ManagedGames()) {
            Record m;
            if (!LoadRecord(RecordPath(dir), m) || m.legacy) {
                if (!m.legacy && DriveUnreachable(dir)) { offline.push_back(dir); LogLine(ctx, L"launcher uninstall: " + dir + L" is on a drive that is not connected: left as it is"); }
                else RemoveManagedGame(dir);
                continue;
            }
            games.push_back({dir, m});
        }
        // a game that kept its own Apps & features entry (a 0.5 install not adopted, an adoption declined at the permission
        // prompt) has WrongGPU too: "from games too" removes it from those as well, in the same batch
        if (removeGames)
            for (auto& dir : RegisteredInstalls()) {
                if (dir.empty() || !IsFile(RecordPath(dir))) continue;
                bool listed = false;
                for (auto& gm : games) listed = listed || IEq(FullPath(gm.first), FullPath(dir));
                Record m;
                if (!listed && LoadRecord(RecordPath(dir), m) && !m.legacy) games.push_back({dir, m});
            }
        if (removeGames)
            for (auto& gm : games)
                if (!gm.second.exe.empty() && GameRunning(gm.second.exe)) {
                    if (why) *why = nameOf(gm.first, gm.second) + L" is running. Close it and try again. Nothing was changed.";
                    return false;
                }
        // all the games' operations handed over at once: the window runs the ones in protected folders in one elevated
        // worker (one permission prompt, not one per game)
        std::vector<OpRequest> todo;
        std::vector<wstring> names;
        for (auto& gm : games) {
            if (!removeGames && !gm.second.regkey.empty()) continue;  // it has its own uninstall entry already
            OpRequest r; r.op = removeGames ? Op::Uninstall : Op::Detach; r.dir = gm.first; r.exe = gm.second.exe;
            todo.push_back(r);
            names.push_back(nameOf(gm.first, gm.second));
        }
        if (sharedStartsAsLink && !todo.empty()) {
            if (why) *why = L"The WrongGPU data folder is a link. Remove or relocate that link before uninstalling games.";
            LogLine(ctx, L"launcher uninstall: refused game operations through a data folder link");
            return false;
        }
        std::vector<OpResult> results;
        if (gameOps) results = gameOps(todo);
        else
            for (auto& r : todo) {
                if (!results.empty() && results.back().code) { OpResult skip; skip.code = kOpNotRun; results.push_back(skip); continue; }
                results.push_back(RunOp(ctx, r, ui));
            }
        while (results.size() < todo.size()) { OpResult skip; skip.code = kOpNotRun; results.push_back(skip); }
        size_t done = 0, failed = todo.size();
        for (size_t i = 0; i < todo.size(); ++i) {
            if (results[i].code == EXIT_OK) {
                for (auto& w : results[i].warnings) ui.log(L"warning: " + w);
                ui.log(removeGames ? L"removed WrongGPU from " + todo[i].dir : L"kept WrongGPU in " + todo[i].dir + L" (its uninstaller is in wronggpu-setup)");
                ++done;
            } else if (results[i].code != kOpNotRun && failed == todo.size()) failed = i;
        }
        for (size_t i = 0; i < todo.size() && failed == todo.size(); ++i)  // none failed, yet one did not run: never go on without it
            if (results[i].code == kOpNotRun) { failed = i; results[i].message = L"It did not run."; }
        if (failed < todo.size()) {
            if (why) {
                *why = (removeGames ? L"WrongGPU could not be uninstalled from " : L"WrongGPU could not get its own uninstaller back in ") + names[failed] + L". " + results[failed].message;
                if (!why->empty() && why->back() != L'.') *why += L".";
                if (removeGames && done) *why += done == 1 ? L" 1 other game has its original files back." : L" " + std::to_wstring(done) + L" other games have their original files back.";
                *why += L" The launcher stays installed; try again.";
            }
            return false;
        }
        // a game on a drive that is not connected keeps WrongGPU as it is (nothing in it can change now): its own Apps & features
        // entry, written from the list alone, so it can still be uninstalled once the drive is back, with no launcher (its
        // restorer is in its wronggpu-setup folder and finds the entry by the folder: UnregisterUninstall)
        for (auto& dir : offline) RegisterUnreachableGame(ctx, dir);
        // 2. The launcher: its entry, its shortcut and its own files. The games' shared files (the model data, the list the
        //    games' uninstallers read, compare.txt) stay while a game uses them.
        const wstring madeAbove = RegString(UninstallHive(), key.c_str(), kMadeFolder);  // read before the entry goes
        RegDeleteKeyW(UninstallHive(), key.c_str());
        const wstring menuLnk = !startMenu.empty() && startMenu != L"-" ? Join(startMenu, L"WrongGPU.lnk") : lnk;
        if (!menuLnk.empty()) DeleteFileW(menuLnk.c_str());
        const wstring base = EnvVar(L"LOCALAPPDATA");
        const wstring shared = base.empty() ? wstring() : Join(base, L"WrongGPU");
        const wstring data = p.custom || !installed || shared.empty() ? p.dataDir : shared;  // the data folder of the launcher that goes
        LogLine(ctx, wstring(L"launcher uninstalled") + (removeGames ? L" with its games" : L"; games kept"));
        {   // from here on nothing is written into the folders that go (a log line would bring them back)
            std::lock_guard<std::mutex> lock(ctx.logMutex);
            if (StartsWithI(ctx.logFile, data + L"\\") || (!shared.empty() && StartsWithI(ctx.logFile, shared + L"\\"))) ctx.logFile.clear();
        }
        const DWORD dataAttrs = GetFileAttributesW(data.c_str());
        const bool dataIsLink = dataAttrs != INVALID_FILE_ATTRIBUTES && (dataAttrs & FILE_ATTRIBUTE_REPARSE_POINT);
        if (!dataIsLink) RemoveLauncherFiles(data);  // never remove named files through a data-folder link
        if (!IEq(FullPath(data), FullPath(shared))) RemoveDirectoryW(data.c_str());  // a portable or --data folder: empty now
        // the shared folder (%LOCALAPPDATA%\WrongGPU) once no game uses it: the model data, the games' list and the logs go too.
        // A game on a drive that is not connected still uses it (it stays on the list, its wronggpu.ini names the model file)
        if (unreachable) *unreachable = offline;
        // a portable copy that uses the shared data keeps it as well (the question said so): its model data is never taken from
        // under it (it is noted in the shared folder while it is there with its marker)
        const bool portableUses = PortableLauncherUsesSharedData();
        if (portableUses) LogLine(ctx, L"launcher uninstall: the shared model data stays (a portable launcher uses it)");
        const bool gamesLeft = AnyGameHasWrongGPU() || !offline.empty() || portableUses;
        const wstring trash = shared.empty() ? wstring() : shared + L".removed-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64());
        bool trashMade = false;
        const DWORD sharedAttrs = shared.empty() ? INVALID_FILE_ATTRIBUTES : GetFileAttributesW(shared.c_str());
        const bool sharedIsLink = sharedAttrs != INVALID_FILE_ATTRIBUTES && (sharedAttrs & FILE_ATTRIBUTE_REPARSE_POINT);
        if (!shared.empty() && !gamesLeft && sharedIsLink) {
            if (!RemoveDirectoryW(shared.c_str())) {
                if (why) *why = L"Could not remove the WrongGPU data folder link: " + ErrText(GetLastError());
                return false;
            }
            LogLine(ctx, L"launcher uninstall: removed the WrongGPU data folder link; its target was kept: " + shared);
        } else if (!shared.empty() && !gamesLeft && IsDir(shared)) {
            // a file held open for a moment (the window's health check hashing the model just after start) lets go soon: the
            // move is tried again for two seconds before what can go goes in place
            for (int i = 0; i < 10 && !trashMade; ++i) {
                if (MoveFileExW(shared.c_str(), trash.c_str(), MOVEFILE_WRITE_THROUGH)) trashMade = true;
                else Sleep(200);
            }
            if (!trashMade) {
                const bool ordinaryShared = RemoveSharedTree(ctx, shared);
                if (!ordinaryShared && IsDir(shared)) { if (why) *why = L"Could not remove the WrongGPU data folder link."; return false; }
                // Only enumerate an ordinary folder; a linked root must never be followed, even after an unlink failure.
                if (ordinaryShared) {
                    std::vector<wstring> left, dirs{shared};
                    for (size_t i = 0; i < dirs.size(); ++i) {
                        for (auto& f : ListFiles(dirs[i], L"*")) left.push_back(f);
                        for (auto& d : ListDirs(dirs[i])) dirs.push_back(d);
                    }
                    for (auto& f : left) { LogLine(ctx, L"launcher uninstall: deleted once free: " + f); DeleteWhenFree(ctx, f, false, Parent(f)); }
                }
            }
        }
        // 3. The installed program: moved out of its folder at once (a running program can be renamed, as this one is when
        //    its own window uninstalls it), the folder removed; the hidden cleanup deletes it once it no longer runs.
        wstring exeDirLeft;  // the install folder when it could not go at once (the hidden cleanup removes it, once empty)
        // the topmost folder above the install folder that the first install made, when the install folder is under it
        const wstring upTo = installed && !madeAbove.empty() && StartsWithI(FullPath(Parent(installedExe)), FullPath(madeAbove) + L"\\") ? madeAbove : wstring();
        if (installed) {
            const wstring exeDir = Parent(installedExe);
            // a launcher started from its Start-menu shortcut runs with its install folder as the current directory, and
            // Windows does not remove a folder a process stands in: this process moves out of it first
            wchar_t cur[MAX_PATH] = {};
            const DWORD cn = GetCurrentDirectoryW(MAX_PATH, cur);
            if (cn > 0 && cn < MAX_PATH) {
                wstring c = cur;
                while (c.size() > 3 && c.back() == L'\\') c.pop_back();
                if (IEq(FullPath(c), FullPath(exeDir)) || StartsWithI(FullPath(c), FullPath(exeDir) + L"\\")) {
                    wchar_t sys[MAX_PATH]; GetSystemDirectoryW(sys, MAX_PATH);
                    SetCurrentDirectoryW(sys);
                }
            }
            bool moved = false;
            if (!trash.empty() && (trashMade || CreateDirectoryW(trash.c_str(), nullptr))) {
                trashMade = true;
                SetFileAttributesW(installedExe.c_str(), FILE_ATTRIBUTE_NORMAL);
                moved = MoveFileExW(installedExe.c_str(), Join(trash, L"WrongGPU.exe").c_str(), MOVEFILE_REPLACE_EXISTING) != FALSE;
                for (auto ext : {L".old", L".new"}) {
                    const wstring f = installedExe + ext;
                    if (IsFile(f) && !DeleteFileW(f.c_str())) MoveFileExW(f.c_str(), Join(trash, BaseName(f)).c_str(), MOVEFILE_REPLACE_EXISTING);
                }
            }
            if (moved || DeleteFileW(installedExe.c_str())) {
                if (!RemoveDirectoryW(exeDir.c_str()) && IsDir(exeDir)) exeDirLeft = exeDir;
                else if (!upTo.empty()) RemoveEmptyUpTo(Parent(exeDir), upTo);  // %LOCALAPPDATA%\Programs it made
            }
            else DeleteWhenFree(ctx, installedExe, false, exeDir, upTo);  // another volume (a --dir install): deleted in place once it closes
        }
        // the install folder that could not go yet goes after the moved exe (rmdir without /s: only while it is empty), and the
        // folders above it that the first install made with it (madeAbove), so no empty %LOCALAPPDATA%\Programs stays
        if (trashMade) {
            const bool ordinaryTrash = RemoveSharedTree(ctx, trash);
            if (!ordinaryTrash && IsDir(trash)) { if (why) *why = L"Could not remove the WrongGPU data folder link."; return false; }
            if (ordinaryTrash && IsDir(trash)) DeleteWhenFree(ctx, trash, true, exeDirLeft, upTo, true);
        }
        if (!exeDirLeft.empty() && !IsDir(trash)) RemoveEmptyUpTo(exeDirLeft, upTo);
        return true;
    } catch (const SetupError& e) { if (why) *why = e.msg; return false; }
}

// ---- Play ------------------------------------------------------------------------------------------------------------
std::vector<wchar_t> GameEnvironment(const std::vector<wstring>& extra) {
    static const wchar_t* const kDrop[] = {L"LLVM_PATH", L"ROCM_PATH", L"HIP_PATH", L"HIP_DEVICE_LIB_PATH"};
    std::vector<wstring> vars;
    if (wchar_t* env = GetEnvironmentStringsW()) {
        for (const wchar_t* p = env; *p; p += wcslen(p) + 1) {
            const wstring v = p;
            const size_t eq = v.find(L'=', 1);
            const wstring name = eq == wstring::npos ? v : v.substr(0, eq);
            bool drop = false;
            for (auto d : kDrop) drop |= IEq(name, d);
            for (auto& x : extra) { const size_t xe = x.find(L'='); drop |= xe != wstring::npos && IEq(name, x.substr(0, xe)); }
            if (!drop) vars.push_back(v);
        }
        FreeEnvironmentStringsW(env);
    }
    for (auto& x : extra) vars.push_back(x);
    std::vector<wchar_t> block;
    for (auto& v : vars) { block.insert(block.end(), v.begin(), v.end()); block.push_back(0); }
    block.push_back(0);
    return block;
}
// The direct start (PlayGame without a Steam app id): the exe with GameEnvironment's environment in its own folder.
bool PlayDirect(const wstring& exe, wstring* why) {
    std::vector<wchar_t> env = GameEnvironment();
    wstring cmd = L"\"" + exe + L"\"";
    std::vector<wchar_t> buf(cmd.begin(), cmd.end()); buf.push_back(0);
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    // a file Windows can't run (not a program, a 16-bit one) fails here with its error, not with Windows' own box ("Unsupported
    // 16-Bit Application") over the launcher while the game shows as starting: no hard-error box on this thread meanwhile. The
    // game gets Windows' default error mode, not this one.
    DWORD oldMode = 0;
    const bool modeSet = SetThreadErrorMode(SEM_FAILCRITICALERRORS | SEM_NOOPENFILEERRORBOX, &oldMode) != 0;
    const BOOL created = CreateProcessW(exe.c_str(), buf.data(), nullptr, nullptr, FALSE, CREATE_UNICODE_ENVIRONMENT | CREATE_DEFAULT_ERROR_MODE, env.data(), Parent(exe).c_str(), &si, &pi);
    const DWORD createErr = created ? 0 : GetLastError();
    if (modeSet) SetThreadErrorMode(oldMode, nullptr);
    if (!created) {
        const DWORD err = createErr;
        if (err == ERROR_ELEVATION_REQUIRED) {
            // a game whose manifest asks for administrator rights: started through the shell, which shows Windows' prompt.
            // The shell gives it this process's environment: the variables GameEnvironment leaves out are left out here too
            // while it starts (and put back after)
            static const wchar_t* const kDrop[] = {L"LLVM_PATH", L"ROCM_PATH", L"HIP_PATH", L"HIP_DEVICE_LIB_PATH"};
            std::vector<std::pair<const wchar_t*, wstring>> kept;
            for (auto n : kDrop) {
                std::vector<wchar_t> val(32768);
                const DWORD len = GetEnvironmentVariableW(n, val.data(), (DWORD)val.size());
                if (len > 0 && len < val.size()) { kept.push_back({n, wstring(val.data(), len)}); SetEnvironmentVariableW(n, nullptr); }
            }
            const wstring dir = Parent(exe);
            SHELLEXECUTEINFOW sei{};
            sei.cbSize = sizeof(sei);
            sei.fMask = SEE_MASK_NOASYNC | SEE_MASK_FLAG_NO_UI;
            sei.lpVerb = L"open";
            sei.lpFile = exe.c_str();
            sei.lpDirectory = dir.c_str();
            sei.nShow = SW_SHOWNORMAL;
            const BOOL ok = ShellExecuteExW(&sei);
            const DWORD serr = ok ? 0 : GetLastError();
            for (auto& k : kept) SetEnvironmentVariableW(k.first, k.second.c_str());
            if (ok) return true;
            if (why) *why = serr == ERROR_CANCELLED ? L"cancelled: the administrator prompt was declined" : L"the game could not be started: " + ErrText(serr);
            return false;
        }
        if (why) *why = L"the game could not be started: " + ErrText(err);
        return false;
    }
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return true;
}
// Steam's installer registers the steam: address (HKEY_CLASSES_ROOT holds the machine's and this user's registrations)
bool SteamAddressKnown() { return !RegString(HKEY_CLASSES_ROOT, L"steam\\shell\\open\\command", nullptr).empty(); }
// the Epic Games Launcher registers com.epicgames.launcher: the same way (rc6 L7)
bool EpicAddressKnown() { return !RegString(HKEY_CLASSES_ROOT, L"com.epicgames.launcher\\shell\\open\\command", nullptr).empty(); }
PlayWay ChoosePlayWay(const wstring& exe, bool steamKnown, bool epicKnown, const wstring& epicApp) {
    PlayWay w;
    w.appid = SteamAppId(exe);
    if (w.appid.empty()) {  // not a Steam game: an Epic one starts through Epic (rc6 L7)
        w.epicApp = epicApp;
        if (w.epicApp.empty()) return w;
        if (!epicKnown) { w.note = L"Windows has no handler for com.epicgames.launcher: addresses"; return w; }
        w.epic = true;
        return w;
    }
    if (!steamKnown) { w.note = L"Windows has no handler for steam: addresses"; return w; }
    w.steam = true;
    return w;
}
// The steam: (or com.epicgames.launcher:) address opened through the shell. A store client it starts (none running yet) gets
// this process's environment: the variables GameEnvironment leaves out are left out while it starts (and put back after), as
// for the direct start.
static bool OpenStoreAddress(const wstring& url, wstring* why) {
    static const wchar_t* const kDrop[] = {L"LLVM_PATH", L"ROCM_PATH", L"HIP_PATH", L"HIP_DEVICE_LIB_PATH"};
    std::vector<std::pair<const wchar_t*, wstring>> kept;
    for (auto n : kDrop) {
        std::vector<wchar_t> val(32768);
        const DWORD len = GetEnvironmentVariableW(n, val.data(), (DWORD)val.size());
        if (len > 0 && len < val.size()) { kept.push_back({n, wstring(val.data(), len)}); SetEnvironmentVariableW(n, nullptr); }
    }
    const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);  // the shell wants COM
    SHELLEXECUTEINFOW sei{};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOASYNC | SEE_MASK_FLAG_NO_UI;
    sei.lpVerb = L"open";
    sei.lpFile = url.c_str();
    sei.nShow = SW_SHOWNORMAL;
    const BOOL ok = ShellExecuteExW(&sei);
    const DWORD err = ok ? 0 : GetLastError();
    if (SUCCEEDED(co)) CoUninitialize();
    for (auto& k : kept) SetEnvironmentVariableW(k.first, k.second.c_str());
    if (!ok && why) *why = ErrText(err);
    return ok != FALSE;
}
bool PlayGame(const wstring& exe, wstring* why, PlayWay* how, Context* log, const PlayStarters& with) {
    if (how) *how = PlayWay{};
    if (!IsFile(exe)) { if (why) *why = L"the game's exe was not found: " + exe; return false; }
    if (GameRunning(exe)) { if (why) *why = BaseName(exe) + L" is running already"; return false; }
    // a Steam game started by its exe can run outside its Steam context (another or no Steam user: other saves, no Steam
    // Cloud); through Steam it starts as from Steam's own Play
    // (an Epic game the same way through Epic, rc6 L7: the manifests are read only for a game without a Steam app id)
    const bool steamKnown = with.steamKnown ? with.steamKnown() : SteamAddressKnown();
    wstring epicApp;
    bool epicKnown = false;
    if (SteamAppId(exe).empty()) {
        epicApp = with.epicApp ? with.epicApp(exe) : EpicAppName(exe);
        if (!epicApp.empty()) epicKnown = with.epicKnown ? with.epicKnown() : EpicAddressKnown();
    }
    PlayWay w = ChoosePlayWay(exe, steamKnown, epicKnown, epicApp);
    wstring err;
    bool ok = false;
    if (w.steam) {
        const wstring url = L"steam://rungameid/" + w.appid;
        ok = with.openUrl ? with.openUrl(url, &err) : OpenStoreAddress(url, &err);
        if (ok) { if (log) LogLine(*log, L"play: " + exe + L" started through Steam (app id " + w.appid + L", " + url + L")"); }
        else { w.steam = false; w.note = L"starting it through Steam failed: " + err; err.clear(); }
    }
    if (w.epic) {
        const wstring url = L"com.epicgames.launcher://apps/" + w.epicApp + L"?action=launch&silent=true";
        ok = with.openUrl ? with.openUrl(url, &err) : OpenStoreAddress(url, &err);
        if (ok) { if (log) LogLine(*log, L"play: " + exe + L" started through Epic (app " + w.epicApp + L", " + url + L")"); }
        else { w.epic = false; w.note = L"starting it through Epic failed: " + err; err.clear(); }
    }
    if (!w.steam && !w.epic) {
        ok = with.direct ? with.direct(exe, &err) : PlayDirect(exe, &err);
        const wstring because = !w.appid.empty() ? L" (Steam app id " + w.appid + L": " + w.note + L")"
                              : !w.epicApp.empty() ? L" (Epic app " + w.epicApp + L": " + w.note + L")" : wstring();
        if (log) LogLine(*log, L"play: " + exe + (ok ? L" started directly" : L" was not started directly: " + err) + because);
    }
    if (how) *how = w;
    if (!ok && why) *why = err;
    return ok;
}

// ---- update check --------------------------------------------------------------------------------------------------
static wstring JsonField(const string& body, const char* name) {
    const string key = string("\"") + name + "\"";
    size_t p = body.find(key);
    if (p == string::npos) return L"";
    p = body.find(':', p + key.size());
    if (p == string::npos) return L"";
    p = body.find('"', p);
    if (p == string::npos) return L"";
    string v;
    for (size_t i = p + 1; i < body.size() && body[i] != '"'; ++i) {
        if (body[i] == '\\' && i + 1 < body.size()) { ++i; if (body[i] == '/' || body[i] == '"' || body[i] == '\\') v += body[i]; continue; }
        v += body[i];
    }
    return FromUtf8(v);
}
bool ParseUpdateReply(const string& body, wstring& version, wstring& page) {
    version.clear(); page.clear();
    if (body.find("\"tag_name\"") != string::npos) {
        version = JsonField(body, "tag_name");
        page = JsonField(body, "html_url");
    } else {
        for (auto& kv : KeyValues(body)) {
            if (kv.first == "version") version = FromUtf8(kv.second);
            else if (kv.first == "page") page = FromUtf8(kv.second);
        }
    }
    while (!version.empty() && (version[0] == L'v' || version[0] == L'V')) version.erase(0, 1);
    if (!page.empty() && Lower(page).rfind(L"https://", 0) != 0) page.clear();
    return !version.empty() && version.size() < 40 && (version[0] >= L'0' && version[0] <= L'9');
}
UpdateInfo CheckForUpdate(const GlobalSettings& gs, const wstring& currentVersion) {
    UpdateInfo u;
    if (!gs.updateCheck) { u.message = L"Check for new releases is off."; return u; }  // the switch's own name (Settings)
    if (gs.updateUrl.empty()) { u.message = L"Check for new releases is on, but there's no release address to check."; return u; }
    if (Lower(gs.updateUrl).rfind(L"https://", 0) != 0) { u.message = L"The release address must start with https://."; return u; }
    URL_COMPONENTS uc{}; uc.dwStructSize = sizeof(uc);
    wchar_t host[256] = {}, path[2048] = {};
    uc.lpszHostName = host; uc.dwHostNameLength = 256; uc.lpszUrlPath = path; uc.dwUrlPathLength = 2048;
    if (!WinHttpCrackUrl(gs.updateUrl.c_str(), 0, 0, &uc)) { u.message = L"The release address can't be read. Check the address, then try again."; return u; }
    HINTERNET s = WinHttpOpen((wstring(L"WrongGPU-launcher/") + currentVersion).c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    string body;
    DWORD status = 0;
    if (s) {
        WinHttpSetTimeouts(s, 5000, 5000, 5000, 10000);
        if (HINTERNET c = WinHttpConnect(s, host, uc.nPort, 0)) {
            if (HINTERNET rq = WinHttpOpenRequest(c, L"GET", path, nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE)) {
                if (WinHttpSendRequest(rq, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) && WinHttpReceiveResponse(rq, nullptr)) {
                    DWORD len = sizeof(status);
                    WinHttpQueryHeaders(rq, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &len, WINHTTP_NO_HEADER_INDEX);
                    char buf[8192];
                    DWORD got = 0;
                    while (body.size() < 65536 && WinHttpReadData(rq, buf, sizeof buf, &got) && got) body.append(buf, got);
                }
                WinHttpCloseHandle(rq);
            }
            WinHttpCloseHandle(c);
        }
        WinHttpCloseHandle(s);
    }
    // no answer at all (offline, the name not found, a timeout, the secure connection refused): the usual cause is the network;
    // an answer without a release (404, a server error, an empty page): the address
    if (status == 0) { u.message = L"WrongGPU couldn't reach the release address. Check your internet connection, then try again."; return u; }
    if (status != 200 || body.empty()) { u.message = L"The release address didn't have a release. Check the address, then try again."; return u; }
    u.checked = true;
    if (!ParseUpdateReply(body, u.version, u.page)) { u.message = L"The release address answered without a version number. Check the address, then try again."; return u; }
    u.newer = VersionLess(currentVersion, u.version);
    u.message = u.newer ? L"WrongGPU " + u.version + L" is available." : L"WrongGPU " + currentVersion + L" is the newest version.";
    return u;
}

}  // namespace wg
