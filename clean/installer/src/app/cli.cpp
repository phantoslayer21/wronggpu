// SPDX-License-Identifier: MIT
// WrongGPU launcher: the command line (build/launcher-design.md section 2 "CLI"). WrongGPU.exe <verb> [game] [options]:
//   scan                          the games the launcher finds (store libraries, --library=<folder>, managed, added)
//   status [game]                 one game or all: status, versions, last run, problems
//   install | repair | update <game>   (--opti=<OptiScaler download>, --model=<nvngx_dlss.dll>, --experimental,
//                                      --switch: replace the player's own OptiScaler / other upscaler mod (backed up),
//                                      --removeold: remove an older WrongGPU install first, --force: no GPU/HIP block,
//                                      --rr-config-fix=on|off: Borderlands 4 / Hogwarts Legacy's Engine.ini question answered)
//   uninstall | rollback | adopt <game>   (adopt --all: every 0.5.x install the scan finds)
//   clean-leftovers | adopt-leftovers <game>   WrongGPU files without a record (hash-matched to our releases only)
//   set <game> key=value ...      per-game settings (written to the game at once); set --global key=value ...
//                                 set <game> opti.<Section>.<Key>=value: an editable key of the game's OptiScaler.ini
//   get <game> [key] | get --global
//   opti <game> [--reset] [--keep]   the OptiScaler tab: locked and editable keys, outside edits; --reset puts the
//                                 editable keys back byte for byte, --keep accepts values OptiScaler saved over ours
//   coexist <game>                the rules table for one game: every tool found, its fingerprint and decision, the
//                                 proxy name OptiScaler would get and why
//   fingerprint <file> ...        what a DLL is (version resource, exports, text; never its name)
//   collect <game> [--out=<zip>]  logs into a zip
//   hipcheck                      the HIP runtime check alone
//   model status | model convert [--model=<dll>]   [--preset=M|L]: the model by name (its converter map and folder; K too in
//                                 a build with Preset K)
//   rr-data status | rr-data convert [--dll=<RR model file>]   the Ray Reconstruction data (shared by the RR games)
//                                 both converts: --ignore=<path> | --ignore=<path> sha256=<hex> (both must match) |
//                                 --ignore=sha256=<hex> (every identical copy), --ignore-list=<file> (repeatable; added
//                                 to the well-known ignore lists, wg_core/convert_ignore.h), --library=<folder> (its sub-folders
//                                 searched as games, this run only)
//   rr-recheck <game> [--run]   what the window's start does: RR's kernels prepared again after a driver / runtime change
//   self-install [--dir=<folder>] [--startmenu=<folder>|-]   the launcher for this user (one Apps & features entry)
//   self-uninstall [--keep-games] [--startmenu=<folder>]   (no window; Apps & features runs "uninstall-launcher", the window's)
//   remove-data                   the shared data (%LOCALAPPDATA%\WrongGPU) removed: a portable copy's way out (no game and no
//                                 installed launcher may use it)
//   play <game>                   start the game (without the ROCm/HIP SDK variables)
//   update-check                  the opt-in release check (no network while it is off or no URL is set)
//   apply --job=<file> --result=<file>   the elevated worker (started by the window with runas); apply --batch=<file>:
//                                 several jobs in one run (one permission prompt; job= / result= lines, stop=1)
//   ui-dump --dump=<file.json> [--shots=<folder>] [--apply=fps|rrconfig|rrdue]   the window's pages painted offscreen, as JSON
//                                 (tests; --apply: the checklist's FPS button, the Engine.ini choices, or the choice waiting for the game's folder, applied for real)
//   ui-render --out-dir=<folder> --fixture=<file> [--size=1180x760] [--scales=1,1.5] [--frames=a,b] [--theme=light|dark]   fixture screenshots (development)
//   version
// Common: --out=<file.json> (the result as JSON; console output of a GUI program is unreliable), --log=<file>,
// --data=<folder> (the launcher's data folder), --noscan (no store libraries). Exit codes: wg_core/constants.h.
// Written for this project.
#include "app/launcher_app.h"

#include <cstdio>

#include "wg_core/json_writer.h"
#include "wg_core/logs.h"
#include "wg_core/model.h"
#include "wg_core/optiedition.h"
#include "wg_core/optiscaler.h"
#include "wg_core/payload.h"
#include "wg_core/pe.h"
#include "wg_core/rr.h"
#include "wg_core/rrconfig.h"
#include "wg_core/rrprep.h"

namespace wg::app {

LauncherState gl;

const int kRestorerResourceId = 960;  // WrongGPU-Restore.exe, embedded by CMake (gen/restore.rc)

void LauncherInit(const wstring& logOverride, bool worker, bool window) {
    gl.paths = ResolvePaths(SelfPath(), gl.dataOverride);
    gl.global = LoadGlobal(gl.paths);
    g.ctx.ownDataDir = gl.paths.dataDir;  // its ignore-files.txt is a well-known list too (convert_ignore.h)
    g.ctx.hipOptional = true;
    g.ctx.hostedLayout = !kDirectRoute;
    g.ctx.perGameEntry = !LauncherInstalled();
    DWORD sz = 0;
    if (const void* d = ResourceData(kRestorerResourceId, sz)) { g.ctx.restorer = d; g.ctx.restorerSize = sz; }
    if (!logOverride.empty()) g.ctx.logFile = logOverride;
    else if (worker) g.ctx.logFile.clear();  // the elevated worker: its job names the player's log (nothing in this account's folders)
    else if (window && !gl.global.firstRunDone && !gl.paths.portable && !gl.paths.custom && !gl.paths.installedCopy && g.ctx.perGameEntry) {
        // a downloaded copy's first start (the first run offers to install it): nothing on this PC changes before that choice,
        // its log included. The lines wait in the temp folder; the choice puts them where it says (SettleFirstRunLog), a
        // window closed without one deletes them (DropFirstRunLog).
        wchar_t t[MAX_PATH + 1] = {};
        wstring tmp(t, GetTempPathW(MAX_PATH + 1, t));
        while (!tmp.empty() && tmp.back() == L'\\') tmp.pop_back();
        gl.firstRunLog = Join(tmp, L"WrongGPU-first-run-" + std::to_wstring(GetCurrentProcessId()) + L".log");
        g.ctx.logFile = gl.firstRunLog;
    } else { EnsureDir(gl.paths.dataDir); g.ctx.logFile = Join(gl.paths.dataDir, L"launcher.log"); }
    if (window) RecoverRrData(g.ctx);  // a Ray Reconstruction data swap that did not finish (nothing else runs yet)
}
void SettleFirstRunLog() {
    if (gl.firstRunLog.empty()) return;
    const wstring temp = gl.firstRunLog, dest = Join(gl.paths.dataDir, L"launcher.log");
    try { EnsureDir(gl.paths.dataDir); } catch (const SetupError& e) { LogLine(g.ctx, L"the log stays in the temp folder: " + e.msg); return; }
    gl.firstRunLog.clear();
    std::lock_guard<std::mutex> lock(g.ctx.logMutex);  // workers may be logging: the old lines first, then theirs
    if (IEq(g.ctx.logFile, temp)) g.ctx.logFile = dest;
    else if (IEq(gl.opLog, temp)) gl.opLog = dest;  // an operation's own log is active: launcher.log behind it moves
    string lines;
    if (ReadAll(temp, lines) && !lines.empty()) {
        HANDLE h = CreateFileW(dest.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) { DWORD w = 0; WriteFile(h, lines.data(), (DWORD)lines.size(), &w, nullptr); CloseHandle(h); }
    }
    DeleteFileW(temp.c_str());
}
void DropFirstRunLog() {
    if (gl.firstRunLog.empty()) return;
    {
        std::lock_guard<std::mutex> lock(g.ctx.logMutex);
        if (IEq(g.ctx.logFile, gl.firstRunLog)) g.ctx.logFile.clear();
    }
    DeleteFileW(gl.firstRunLog.c_str());
    gl.firstRunLog.clear();
}
// The operation's own log (logs\last-operation.log, folded into launcher.log by OpLogEnd). It cannot be made (a full drive,
// a log viewer holding the file without delete sharing): the operation goes on, its lines in launcher.log; never a throw on
// the operation's worker thread.
void OpLogBegin() {
    if (!gl.opLog.empty()) return;
    wstring opFile;
    try {
        const wstring dir = Join(gl.paths.dataDir, L"logs");
        EnsureDir(dir);
        opFile = Join(dir, L"last-operation.log");
        WriteFileAtomic(opFile, "", 0);
    } catch (const SetupError& e) {
        LogLine(g.ctx, L"the operation's own log could not be started (its lines stay in this log): " + e.msg);
        return;
    }
    gl.opLog = opFile;
    std::lock_guard<std::mutex> lock(g.ctx.logMutex);
    std::swap(g.ctx.logFile, gl.opLog);  // gl.opLog now holds launcher.log
}
void OpLogEnd() {
    if (gl.opLog.empty()) return;
    wstring opFile;
    {
        std::lock_guard<std::mutex> lock(g.ctx.logMutex);
        opFile = g.ctx.logFile;
        g.ctx.logFile = gl.opLog;
    }
    gl.opLog.clear();
    string text;
    if (ReadAll(opFile, text) && !text.empty()) {
        HANDLE h = CreateFileW(g.ctx.logFile.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) { DWORD w; WriteFile(h, text.data(), (DWORD)text.size(), &w, nullptr); CloseHandle(h); }
    }
}

// The elevated worker started with runas for `args`, waited for; false: Windows did not start it (permission not given).
// Called on a worker thread without a message loop: COM is set up for the shell on it, and the shell finishes before it
// returns (SEE_MASK_NOASYNC), as ShellExecuteEx asks of such a thread.
static bool StartElevated(const wstring& args, HWND owner, DWORD& code) {
    const wstring self = SelfPath();
#ifdef WRONGGPU_TESTING
    // tests (r2c): every permission prompt the window would raise is written to WRONGGPU_TEST_ELEVATION_LOG. With protected
    // folders set (WRONGGPU_TEST_PROTECTED, NeedsElevation) no prompt is raised: the worker runs here without elevation as the
    // granted prompt's stand-in (WRONGGPU_TEST_ELEVATED=1: its folders count as writable), or WRONGGPU_TEST_ELEVATION=deny
    // answers No.
    const wstring askLog = EnvVar(L"WRONGGPU_TEST_ELEVATION_LOG");
    if (!askLog.empty()) {
        const string line = ToUtf8(L"runas " + args) + "\r\n";
        HANDLE h = CreateFileW(askLog.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) { DWORD w; WriteFile(h, line.data(), (DWORD)line.size(), &w, nullptr); CloseHandle(h); }
    }
    if (!EnvVar(L"WRONGGPU_TEST_PROTECTED").empty()) {
        (void)owner;
        if (EnvVar(L"WRONGGPU_TEST_ELEVATION") == L"deny") return false;
        std::wstring env;  // this process's environment and WRONGGPU_TEST_ELEVATED=1
        if (wchar_t* block = GetEnvironmentStringsW()) {
            for (const wchar_t* p = block; *p; p += wcslen(p) + 1) env.append(p).push_back(L'\0');
            FreeEnvironmentStringsW(block);
        }
        env.append(L"WRONGGPU_TEST_ELEVATED=1").push_back(L'\0');
        env.push_back(L'\0');
        wstring cmd = L"\"" + self + L"\" " + args;
        STARTUPINFOW si{}; si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        if (!CreateProcessW(self.c_str(), &cmd[0], nullptr, nullptr, FALSE, CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW, &env[0], nullptr, &si, &pi)) return false;
        WaitForSingleObject(pi.hProcess, INFINITE);
        code = 0; GetExitCodeProcess(pi.hProcess, &code);
        CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
        return true;
    }
#endif
    const HRESULT com = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    SHELLEXECUTEINFOW sei{}; sei.cbSize = sizeof(sei); sei.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC; sei.lpVerb = L"runas";
    sei.lpFile = self.c_str(); sei.lpParameters = args.c_str(); sei.nShow = SW_HIDE; sei.hwnd = owner;
    const bool started = ShellExecuteExW(&sei) && sei.hProcess;
    if (SUCCEEDED(com)) CoUninitialize();
    if (!started) return false;
    WaitForSingleObject(sei.hProcess, INFINITE);
    code = 0; GetExitCodeProcess(sei.hProcess, &code);
    CloseHandle(sei.hProcess);
    return true;
}
static const wchar_t* const kNotGranted = L"Administrator permission is needed to change this game folder and was not granted.";

OpResult RunElevated(const OpRequest& r, HWND owner) {
    OpResult res;
    const wstring dir = Join(gl.paths.dataDir, L"jobs");
    EnsureDir(dir);
    const wstring stamp = NowStamp(true);
    const wstring job = Join(dir, L"job-" + stamp + L".txt"), result = Join(dir, L"result-" + stamp + L".txt");
    const string text = JobText(r, EnvVar(L"LOCALAPPDATA"), Join(Join(gl.paths.dataDir, L"logs"), L"elevated-" + stamp + L".log"), CurrentUserSid(), g.ctx.perGameEntry ? 1 : 0,
                              PlayerDocumentsDir(g.ctx));
    WriteFileAtomic(job, text.data(), text.size());
    DWORD code = 0;
    if (!StartElevated(L"apply \"--job=" + job + L"\" \"--result=" + result + L"\"", owner, code)) {
        DeleteFileW(job.c_str());
        res.code = EXIT_REFUSED;
        res.message = kNotGranted;
        return res;
    }
    string out;
    if (!ReadAll(result, out) || !ParseResult(out, res)) { res.code = code ? (int)code : EXIT_FAILED; res.message = L"The elevated worker gave no result (exit " + std::to_wstring(code) + L")."; }
    DeleteFileW(job.c_str()); DeleteFileW(result.c_str());
    return res;
}

// Several operations in one elevated worker (one permission prompt): "apply --batch=<file>" runs the jobs in order; stop:
// none after the first failure. A result per request (kOpNotRun: not run).
static std::vector<OpResult> RunElevatedBatch(const std::vector<OpRequest>& rs, HWND owner, bool stop) {
    std::vector<OpResult> out(rs.size());
    if (rs.empty()) return out;
    const wstring dir = Join(gl.paths.dataDir, L"jobs");
    EnsureDir(dir);
    const wstring stamp = NowStamp(true);
    const wstring log = Join(Join(gl.paths.dataDir, L"logs"), L"elevated-" + stamp + L".log");
    std::vector<std::pair<wstring, wstring>> files;
    for (size_t i = 0; i < rs.size(); ++i) {
        const wstring n = std::to_wstring(i + 1);
        files.push_back({Join(dir, L"job-" + stamp + L"-" + n + L".txt"), Join(dir, L"result-" + stamp + L"-" + n + L".txt")});
        const string text = JobText(rs[i], EnvVar(L"LOCALAPPDATA"), log, CurrentUserSid(), g.ctx.perGameEntry ? 1 : 0,
                              PlayerDocumentsDir(g.ctx));
        WriteFileAtomic(files[i].first, text.data(), text.size());
    }
    const wstring batch = Join(dir, L"batch-" + stamp + L".txt");
    const string list = BatchText(files, stop);
    WriteFileAtomic(batch, list.data(), list.size());
    DWORD code = 0;
    const bool started = StartElevated(L"apply \"--batch=" + batch + L"\"", owner, code);
    bool failed = false;
    for (size_t i = 0; i < rs.size(); ++i) {
        string text;
        if (!started) { out[i].code = EXIT_REFUSED; out[i].message = kNotGranted; }
        else if (!ReadAll(files[i].second, text) || !ParseResult(text, out[i])) {
            out[i] = OpResult();
            if (stop && failed) out[i].code = kOpNotRun;
            else { out[i].code = code ? (int)code : EXIT_FAILED; out[i].message = L"The elevated worker gave no result (exit " + std::to_wstring(code) + L")."; }
        }
        failed = failed || (out[i].code != EXIT_OK && out[i].code != kOpNotRun);
        DeleteFileW(files[i].first.c_str()); DeleteFileW(files[i].second.c_str());
    }
    DeleteFileW(batch.c_str());
    return out;
}

std::vector<OpResult> RunElevatedOps(const std::vector<OpRequest>& rs, HWND owner, bool stop) { return RunElevatedBatch(rs, owner, stop); }

// The per-game operations of the launcher's own install or uninstall, in order: in this process where the folder allows
// it, then every one in a protected folder together in one elevated worker (one permission prompt, not one per game) when
// a window owns the prompt. stop: nothing runs after the first failure (the uninstall). A result per request.
static std::vector<OpResult> GameOpsFor(const std::vector<OpRequest>& rs, HWND elevateOwner, bool stop) {
    std::vector<OpResult> out(rs.size());
    std::vector<size_t> protectedOnes;
    Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
    bool failed = false;
    for (size_t i = 0; i < rs.size(); ++i) {
        if (elevateOwner && !rs[i].dir.empty() && NeedsElevation(rs[i].dir)) { protectedOnes.push_back(i); continue; }
        if (stop && failed) { out[i].code = kOpNotRun; continue; }
        out[i] = RunOp(g.ctx, rs[i], ui);
        failed = failed || out[i].code != EXIT_OK;
    }
    if (protectedOnes.empty()) return out;
    if (stop && failed) { for (size_t i : protectedOnes) out[i].code = kOpNotRun; return out; }
    std::vector<OpRequest> e;
    for (size_t i : protectedOnes) e.push_back(rs[i]);
    const std::vector<OpResult> er = RunElevatedBatch(e, elevateOwner, stop);
    for (size_t k = 0; k < protectedOnes.size(); ++k) out[protectedOnes[k]] = er[k];
    return out;
}

void AdoptGames(const std::vector<wstring>& dirs, HWND elevateOwner, std::vector<wstring>* kept, std::vector<OpResult>* results) {
    std::vector<OpRequest> rs;
    for (auto& d : dirs) { OpRequest r; r.op = Op::Adopt; r.dir = d; rs.push_back(r); }
    const std::vector<OpResult> res = GameOpsFor(rs, elevateOwner, false);
    for (size_t i = 0; i < rs.size(); ++i) {
        if (!res[i].code) continue;
        LogLine(g.ctx, L"adopt: " + rs[i].dir + L" was not adopted: " + res[i].message);
        if (kept) { Record m; kept->push_back(LoadRecord(RecordPath(rs[i].dir), m) && !m.exe.empty() ? GameDisplayName(m.exe) : rs[i].dir); }
    }
    if (results) *results = res;
}

void SyncRrSettings(HWND elevateOwner, const wstring& except, std::vector<wstring>& warnings, const std::vector<RrSettingsFailure>* known) {
    try {
        const std::vector<RrSettingsFailure> failed = known ? *known : RefreshAllRrSettings(g.ctx, except);
        if (failed.empty()) return;
        // the ones this process may not write: an Apply (it writes the RR settings file from the game's profile), the games in
        // protected folders together in one elevated worker (one permission prompt) when a window owns the prompt
        std::vector<OpRequest> rs;
        for (auto& f : failed) { OpRequest r; r.op = Op::Apply; r.dir = f.dir; rs.push_back(r); }
        const std::vector<OpResult> res = elevateOwner ? GameOpsFor(rs, elevateOwner, false) : std::vector<OpResult>();
        for (size_t i = 0; i < failed.size(); ++i) {
            if (i < res.size() && res[i].code == EXIT_OK) { LogLine(g.ctx, L"RR settings of " + failed[i].dir + L" written with permission"); continue; }
            Record m;
            const wstring name = LoadRecord(RecordPath(failed[i].dir), m) && !m.exe.empty() ? GameDisplayName(m.exe) : BaseName(failed[i].dir);
            wstring cause = i < res.size() && !res[i].message.empty() ? res[i].message : failed[i].why;
            while (!cause.empty() && cause.back() == L'.') cause.pop_back();
            warnings.push_back(kRrNotUpdated + name + L" (" + failed[i].dir + L") couldn't be written: " + cause + L". Its Ray Reconstruction stays off until it is repaired.");
        }
    } catch (const SetupError& e) {  // the elevated job could not be written (a full drive): said, never a throw out of here
        warnings.push_back(L"Ray Reconstruction is ready, but the other RR games' settings couldn't be written (" + e.msg + L"). Their Ray Reconstruction stays off until they are repaired.");
    }
}

bool InstallLauncherForUser(const wstring& dir, const wstring& startMenu, HWND elevateOwner, const GlobalSettings& settings, const std::vector<wstring>& alsoAdopt,
                            wstring& why, std::vector<wstring>* kept) {
    if (!SelfInstall(g.ctx, SelfPath(), dir, startMenu, &why)) return false;
    g.ctx.perGameEntry = false;
    // one entry for all: the managed games' own entries go, and the 0.5 installs the window found are adopted with them
    std::vector<wstring> dirs;
    auto add = [&dirs](const wstring& d) {
        if (d.empty() || !IsFile(RecordPath(d))) return;
        for (auto& x : dirs) if (IEq(FullPath(x), FullPath(d))) return;
        dirs.push_back(d);
    };
    for (auto& d : ManagedGames()) add(d);
    for (auto& d : alsoAdopt) add(d);
    std::vector<wstring> notAdopted;
    AdoptGames(dirs, elevateOwner, &notAdopted);
    for (auto& n : notAdopted) LogLine(g.ctx, L"self-install: " + n + L" keeps its own Apps & features entry");
    if (kept) kept->insert(kept->end(), notAdopted.begin(), notAdopted.end());
    // the settings go with it: the installed copy's data folder gets launcher.ini (a portable copy's folders, games and
    // choices carry over, and an uninstaller then keeps the launcher's shared folder). `settings` is the caller's copy: the
    // window stays live meanwhile and may change its own.
    try { SaveGlobal(ResolvePaths(Join(dir.empty() ? DefaultInstallDir() : FullPath(dir), L"WrongGPU.exe"), gl.dataOverride), settings); }
    catch (const SetupError& e) { LogLine(g.ctx, L"self-install: the settings were not copied to the installed launcher: " + e.msg); }
    return true;
}

bool UninstallLauncher(bool removeGames, const wstring& startMenu, HWND elevateOwner, Ui& ui, wstring& why, std::vector<wstring>* unreachable) {
    const bool ok = SelfUninstall(g.ctx, gl.paths, removeGames, startMenu, ui, &why,
                                  [elevateOwner](const std::vector<OpRequest>& rs) { return GameOpsFor(rs, elevateOwner, true); }, unreachable);
    RefreshPerGameEntry();  // a portable window stays open: its next operations give each game its own entry again
    return ok;
}
void RefreshPerGameEntry() { g.ctx.perGameEntry = !LauncherInstalled(); }

namespace {

struct Args {
    std::vector<wstring> pos;            // positional arguments after the verb
    std::map<wstring, wstring> opt;      // --name=value (and --name value for the path options)
    std::set<wstring> flags;             // --name
    std::vector<wstring> libraries;      // --library=<folder> (repeatable)
    std::vector<wstring> ignores;        // --ignore=<path, or sha256=<hex>> (repeatable; convert_ignore.h)
    std::vector<wstring> ignoreLists;    // --ignore-list=<file> (repeatable)
    bool has(const wchar_t* f) const { return flags.count(f) > 0; }
    wstring get(const wchar_t* k) const { auto it = opt.find(k); return it == opt.end() ? wstring() : it->second; }
};

// Console output of a GUI-subsystem program: the inherited handle (redirected), else the parent's console.
void Print(const wstring& s) {
    static HANDLE out = [] {
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        if (h && h != INVALID_HANDLE_VALUE && GetFileType(h) != FILE_TYPE_UNKNOWN) return h;
        if (AttachConsole(ATTACH_PARENT_PROCESS))
            return CreateFileW(L"CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
        return INVALID_HANDLE_VALUE;
    }();
    if (out == INVALID_HANDLE_VALUE) return;
    const string u = ToUtf8(s + L"\r\n");
    DWORD w = 0;
    if (GetFileType(out) == FILE_TYPE_CHAR) WriteConsoleW(out, (s + L"\r\n").c_str(), (DWORD)(s.size() + 2), &w, nullptr);
    else WriteFile(out, u.data(), (DWORD)u.size(), &w, nullptr);
}

// The OptiScaler of an install / update / repair: --opti= when given; else WrongGPU's edition of OptiScaler (optiedition.h:
// the launcher's cached copy, or downloaded now); offline or when that fails, the player's own download (Settings).
wstring OptiFor(const Args& a, const GlobalSettings& gs) {
    if (!a.get(L"opti").empty()) return a.get(L"opti");
    const EditionResult ed = GetEdition(g.ctx, gl.paths.dataDir, [](unsigned long long d, unsigned long long t) {
        static int last = -1;
        const int tenth = t ? (int)(d * 10 / t) : 0;
        if (tenth != last && d) { last = tenth; Print(L"Downloading OptiScaler (WrongGPU edition): " + std::to_wstring(tenth * 10) + L"%"); }
        return true;
    });
    if (!ed.path.empty()) return ed.path;
    if (!gs.optiPackage.empty()) Print(ed.error + L" Using your own OptiScaler download: " + gs.optiPackage);
    return gs.optiPackage;
}

struct Output {
    Json j;
    wstring outFile;
    int code = EXIT_OK;
    int finish() {
        j.N("exit_code", code).End();
        if (!outFile.empty()) { const string s = j.str() + "\n"; try { WriteFileAtomic(outFile, s.data(), s.size()); } catch (const SetupError&) {} }
        return code;
    }
};

void FindingsJson(Json& j, const std::vector<Finding>& findings) {
    j.Key("coexistence").Arr();
    for (auto& f : findings) {
        j.Obj().S("tool", ToolKey(f.tool)).S("decision", DecisionName(f.decision)).S("what", f.what).S("evidence", f.evidence).S("message", f.message);
        if (f.code) j.N("code", f.code);
        j.Key("files").Arr(); for (auto& x : f.files) j.Str(x); j.End();
        j.End();
    }
    j.End();
}

void EntryJson(Json& j, const GameEntry& e) {
    j.Obj().S("name", e.name).S("exe", e.exe).S("dir", e.dir).S("source", e.source).S("status", StatusName(e.status)).S("status_text", e.statusText);
    if (e.hasRecord) {
        j.S("installed_version", e.installedVersion).B("other_build", e.otherBuild).B("record_v1", e.recordV1).B("per_game_entry", e.perGameEntry).B("optiscaler_route", e.hosted)
            .B("adoptable", e.adoptable).B("repair_suggested", e.repairSuggested).S("proxy", e.proxy);
        j.B("can_roll_back", e.hasPrevious);  // the kept version fits this install's layout (launcher.cpp PreviousFits)
        if (e.hasPrevious && !e.previous.empty()) j.S("previous_version", e.previous);
        j.Key("problems").Arr(); for (auto& p : e.problems) j.Str(p); j.End();
        j.Key("last_run").Obj().B("found", e.last.found).S("summary", e.last.Summary()).S("upscaler", e.last.upscaler).S("executor", e.last.executor)
            .B("runtime_ready", e.last.ready).B("optiscaler_plugin", e.last.optiPlugin).S("support_code", e.last.support).S("when", e.last.when)
            .B("upscaled", e.last.upscaled).B("rr_only", e.last.rrOnly).B("rr_upscaled", e.last.rrUpscaled);
        j.Key("notes").Arr(); for (auto& n : e.last.notes) j.Str(n); j.End();
        j.S("plugin", e.last.plugin);
        j.Key("warnings").Arr(); for (auto& w : e.last.warnings) j.Obj().S("kind", w.kind).S("text", w.text).End(); j.End();
        j.Key("rr").Obj().B("found", e.last.rrFound).B("used", e.last.rrUsed).S("mode", e.last.rrMode).S("ms", e.last.rrMs).N("frames", e.last.rrFrames).End();
        j.End();
    }
    // Ray Reconstruction (rr.h): only a game that uses it says anything of it
    j.B("rr_uses", e.rrUses);
    if (e.rrUses) j.S("rr_why", e.rrWhy).B("rr_installed", e.rrInstalled).B("rr_enabled", e.rrEnabled).S("rr_mode", e.rrMode).B("rr_path_tracing_only", e.rrPathTracingOnly);
    // a game that creates RR itself (rr.h RrUse::direct): RR is available only where the install's OptiScaler is WrongGPU's
    // edition (an install with another OptiScaler: rr_available false, with the plain reason)
    if (e.rrUses) {
        const bool noEdition = e.rrDirect && e.hasRecord && e.rrInstalled && !e.optiEdition;
        j.B("rr_direct", e.rrDirect).B("rr_available", !noEdition);
        if (e.hasRecord) j.B("optiscaler_edition", e.optiEdition);
        if (noEdition) j.S("rr_unavailable_why", L"the game creates Ray Reconstruction itself, and this install's OptiScaler is not WrongGPU's edition (plugin interface 4 or newer)");
    }
    // a tool of the player's the game needs first (gamerules.h): only such a game says anything of it
    if (!e.needs.empty()) j.Key("needs").Obj().S("tool", e.needs).B("found", e.needsFound).S("evidence", e.needsWhy).S("page", e.needsPage).End();
    j.Key("found").Arr(); for (auto& f : e.found) j.Str(f); j.End();
    FindingsJson(j, e.coexist);
    j.Key("proxy_plan").Obj().S("name", e.proxyPlan.name).S("reason", e.proxyPlan.reason).B("untested", e.proxyPlan.untested).End();
    if (!e.leftovers.empty()) {
        j.Key("leftovers").Arr();
        for (auto& l : e.leftovers) j.Obj().S("path", l.rel).S("sha256", l.sha).S("version", l.version).End();
        j.End();
    }
    j.End();
}

void ResultJson(Output& o, const OpResult& r) {
    o.code = r.code;
    o.j.S("message", r.message);
    o.j.Key("warnings").Arr(); for (auto& w : r.warnings) o.j.Str(w); o.j.End();
    o.j.Key("notes").Arr(); for (auto& n : r.notes) o.j.Str(n); o.j.End();
    Print(r.message);
    for (auto& w : r.warnings) Print(L"  warning: " + w);
    for (auto& n : r.notes) Print(L"  " + n);
}

bool Game(const Args& a, Output& o, wstring& exe, wstring& dir) {
    if (a.pos.empty()) { o.code = EXIT_USAGE; o.j.S("error", L"name the game: its .exe, its folder or its install folder"); Print(L"name the game: its .exe, its folder or its install folder"); return false; }
    if (!ResolveGameArg(g.ctx, a.pos[0], exe, dir)) { o.code = EXIT_REFUSED; o.j.S("error", L"no game found at " + a.pos[0]); Print(L"no game found at " + a.pos[0]); return false; }
    return true;
}

// Borderlands 4, Hogwarts Legacy (rrconfig.h) after an install, repair or update: --rr-config-fix=on|off answers the window's
// question from the command line; without it, a game that has no answer yet gets the question, the steps and how to answer
// in its result (rr_config_question), as the window asks it.
static void RrConfigAfterInstall(const Args& a, const wstring& dir, OpResult& res, Output& o) {
    Record m;
    const RrConfigGame* rg = LoadRecord(RecordPath(dir), m) ? RrConfigGameFor(m.exe) : nullptr;
    if (!rg) return;
    if (DetectRr(m.exe).notOffered) return;  // rc6: no question where WrongGPU's RR isn't offered (the updated Hogwarts Legacy, rr.cpp)
    Profile p = GameProfile(dir);
    const wstring choice = a.get(L"rr-config-fix");
    if (!choice.empty()) {
        wstring why;
        if (!SetSetting(p, "rr_config_fix", ToUtf8(choice), &why)) { res.warnings.push_back(L"--rr-config-fix: " + why); return; }
        OpRequest r; r.op = Op::Apply; r.dir = dir; r.hasProfile = true; r.profile = p;
        Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
        const OpResult ar = RunOp(g.ctx, r, ui);
        for (auto& w : ar.warnings) res.warnings.push_back(w);
        for (auto& n : ar.notes) res.notes.push_back(n);
        if (ar.code) res.warnings.push_back(L"rr_config_fix was not applied: " + ar.message);
        return;
    }
    if (p.get("rr_config_fix") != "ask") return;
    const wstring steps = RrConfigSteps(*rg, RrConfigFile(g.ctx, *rg, nullptr));
    const wstring set = L"WrongGPU.exe set \"" + m.exe + L"\" rr_config_fix=on";
    o.j.Key("rr_config_question").Obj().S("game", rg->name).S("text", RrConfigNotice(*rg)).S("steps", steps).S("fix", set)
        .S("myself", L"WrongGPU.exe set \"" + m.exe + L"\" rr_config_fix=off").End();
    Print(RrConfigNotice(*rg) + L"\n\nFix it for me: " + set + L"\nDo it yourself: the steps below (rr_config_fix=off stops this question).\n\n" + steps);
}

// model convert / rr-data convert: the store libraries (unless --noscan) and the sub-folders of each --library=<folder> (one
// game each, as `scan` reads them; this run only).
static std::vector<StoreGame> ConvertLibraries(const Args& a, bool noScan) {
    std::vector<StoreGame> libs = noScan ? std::vector<StoreGame>{} : ScanLibraries();
    for (auto& lib : a.libraries)
        for (auto& sub : ListDirs(FullPath(lib))) libs.push_back({BaseName(sub), sub, L"", L"Folder"});
    return libs;
}
// --ignore-list=<file> and --ignore=<entry> of model convert / rr-data convert (convert_ignore.h), added to the well-known
// lists. A list file that can't be read, or an --ignore= that is not understood, stops the command before anything is
// converted (a file meant to be skipped is never converted because of a typo); a bad line inside a list is skipped with a warning.
static bool ConvertIgnoreArgs(const Args& a, Output& o) {
    g.ctx.ignoreListFiles.clear();
    g.ctx.ignoreListBytes.clear();
    g.ctx.ignoreEntries.clear();
    auto refuse = [&o](const wstring& msg) { o.code = EXIT_USAGE; o.j.S("error", msg); Print(msg); return false; };
    // a bare --ignore / --ignore-list (no "=", or nothing after it): never takes the next argument as its value
    if (a.has(L"ignore")) return refuse(L"--ignore needs its entry after an equals sign: --ignore=<full path> or --ignore=sha256=<hex>; nothing was converted");
    if (a.has(L"ignore-list")) return refuse(L"--ignore-list needs its file: --ignore-list=<file>; nothing was converted");
    for (auto& f : a.ignoreLists) {
        // read once, with the lists' own bounded reader (the size is checked before anything is read); the search parses
        // these bytes (LoadConvertIgnore), it does not read the file again
        const wstring full = FullPath(f);
        string bytes;
        wstring why;
        if (!ReadIgnoreListBytes(full, bytes, why)) return refuse(L"--ignore-list=" + f + L": the file can't be read (" + full + L": " + why + L"); nothing was converted");
        g.ctx.ignoreListFiles.push_back(full);
        g.ctx.ignoreListBytes.push_back(std::move(bytes));
    }
    for (auto& s : a.ignores) {
        IgnoreEntry e;
        wstring why;
        // a hash alone (--ignore=sha256=<hex>) is the player's own choice here; it matches every copy of that file with the same bytes
        if (!ParseIgnoreLine(s, e, why, nullptr, true) || e.text.empty())
            return refuse(L"--ignore=" + s + L": " + (why.empty() ? wstring(L"no path or sha256=<hex>") : why) + L"; nothing was converted");
        g.ctx.ignoreEntries.push_back(s);
    }
    return true;
}
// --ignore and --ignore-list are read by model convert and rr-data convert / prepare only: on any other verb they would be
// accepted and do nothing, so the player thinks a file is protected. Exit 10 and a message instead.
static bool IgnoreArgsMisplaced(const wstring& verb, const Args& a, Output& o) {
    if (a.ignores.empty() && a.ignoreLists.empty() && !a.has(L"ignore") && !a.has(L"ignore-list")) return false;
    const wstring sub = a.pos.empty() ? L"status" : a.pos[0];
    bool applies = (verb == L"model" && sub == L"convert") || (verb == L"rr-data" && sub == L"convert");
#ifdef WRONGGPU_TESTING
    applies = applies || (verb == L"rr-data" && sub == L"prepare");
#endif
    if (applies) return false;
    const wstring msg = L"--ignore and --ignore-list apply only to \"model convert\" and \"rr-data convert\" (the well-known ignore lists are read by every conversion, "
                        L"also an install's); \"" + verb + (a.pos.empty() ? wstring() : L" " + a.pos[0]) + L"\" would ignore them, so it was not run";
    o.code = EXIT_USAGE;
    o.j.S("error", msg);
    Print(msg);
    return true;
}
// The files the ignore lists left out, with the reason, and the lists' warnings: in the result and on the console.
static void IgnoreJson(Output& o, const IgnoreReport& r) {
    o.j.Key("skipped").Arr();
    for (auto& s : r.skipped) { o.j.Str(s); Print(L"  skipped: " + s); }
    o.j.End();
    o.j.Key("ignore_warnings").Arr();
    for (auto& w : r.warnings) { o.j.Str(w); Print(L"  ignore list warning: " + w); }
    o.j.End();
}

int Verb(const wstring& verb, const Args& a, Output& o) {
    if (IgnoreArgsMisplaced(verb, a, o)) return o.code;
    const bool noScan = a.has(L"noscan");
    GlobalSettings& gs = gl.global;
    GlobalSettings scan = gs;  // --library=<folder>: scanned this time only, never saved
    for (auto& lib : a.libraries) scan.extraFolders.push_back(FullPath(lib));
    if (verb == L"version") {
        o.j.S("build", g.ctx.data->buildId).S("payload", g.ctx.data->payloadName);
        // rc5: the link time that tells two builds of the same version apart (DecideLauncherStart)
        const uint32_t built = FileLinkTime(SelfPath());
        o.j.N("built_unix", (long long)built).S("built", BuildTimeText(built));
        // every file the launcher embeds (release checks: never OptiScaler, which the launcher downloads, optiedition.h)
        o.j.Key("embedded").Arr();
        for (size_t i = 0; i < g.ctx.data->count; ++i) o.j.Str(g.ctx.data->files[i].rel);
        for (size_t i = 0; i < g.ctx.data->rrCount; ++i) o.j.Str(wstring(L"rr\\") + g.ctx.data->rrFiles[i].rel);
        o.j.End();
        Print(wstring(L"WrongGPU ") + g.ctx.data->version + L" (build " + g.ctx.data->buildId + L", payload " + g.ctx.data->payloadName + L")");
        return EXIT_OK;
    }
    if (verb == L"scan" || (verb == L"status" && a.pos.empty())) {
        const auto games = ScanGames(g.ctx, scan, noScan);
        o.j.Key("games").Arr();
        for (auto& e : games) { EntryJson(o.j, e); Print(e.name + L"  [" + e.statusText + L"]  " + e.exe); }
        o.j.End();
        return EXIT_OK;
    }
    if (verb == L"status") {
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        GameEntry e = EntryFor(g.ctx, exe.empty() ? Join(dir, L"?.exe") : exe, L"", L"Command line");
        o.j.Key("game"); EntryJson(o.j, e);
        Print(e.name + L": " + e.statusText);
        if (e.hasRecord) Print(L"  last run: " + e.last.Summary());
        return EXIT_OK;
    }
    if (verb == L"fingerprint") {
        if (a.pos.empty()) { o.j.S("error", L"name one or more files"); return EXIT_USAGE; }
        o.j.Key("files").Arr();
        for (auto& p : a.pos) {
            const Fingerprint f = IdentifyFile(FullPath(p));
            o.j.Obj().S("file", FullPath(p)).S("tool", ToolKey(f.tool)).S("name", ToolName(f.tool)).B("pe", f.pe).S("version", f.version)
                .S("company", f.company).S("product", f.product).S("evidence", f.evidence).End();
            Print(p + L": " + f.Describe());
        }
        o.j.End();
        return EXIT_OK;
    }
    if (verb == L"coexist") {
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        Record m;
        const bool rec = LoadRecord(RecordPath(dir), m) && !m.legacy;
        const CoexistReport r = AssessCoexistence(g.ctx, exe, dir, rec ? &m : nullptr);
        o.j.S("exe", exe).S("dir", dir);
        FindingsJson(o.j, r.findings);
        o.j.Key("proxy").Obj().S("name", r.proxy.name).S("reason", r.proxy.reason).B("untested", r.proxy.untested).End();
        o.j.B("asi_loader", r.asiLoader).B("hip_tool", r.hipTool).S("players_optiscaler_proxy", r.playersOptiProxy);
        o.j.Key("graphics").Obj().B("pe", r.api.pe).B("d3d12", r.api.d3d12).S("evidence", r.api.evidence).Key("others").Arr();
        for (auto& x : r.api.others) o.j.Str(x);
        o.j.End().End();
        if (const Finding* no = r.refusal()) o.j.S("refused", no->what);
        for (auto& f : r.findings) Print(wstring(DecisionName(f.decision)) + L": " + f.what + L" - " + f.message);
        Print(L"OptiScaler's name: " + (r.proxy.name.empty() ? wstring(L"(none free)") : r.proxy.name) + L" - " + r.proxy.reason);
        return EXIT_OK;
    }
    if (verb == L"opti") {
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        Record m;
        if (!LoadRecord(RecordPath(dir), m) || m.legacy) { o.j.S("error", L"WrongGPU is not installed in " + dir); Print(L"WrongGPU is not installed in " + dir); return EXIT_NOT_INSTALLED; }
        if (a.has(L"reset") || a.has(L"keep")) {
            if (NeedsElevation(dir)) { o.j.S("error", L"administrator permission is needed for " + dir); return EXIT_REFUSED; }
            OpLogBegin();
            OpRequest r; r.op = Op::Apply; r.dir = dir; r.optiReset = a.has(L"reset"); r.optiKeep = a.has(L"keep");
            r.optiOnly = true;  // the tab only: wronggpu.ini and the profile's keys stay as they are
            Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
            const OpResult res = RunOp(g.ctx, r, ui);
            OpLogEnd();
            ResultJson(o, res);
            if (res.code) return res.code;
            m = Record();
            LoadRecord(RecordPath(dir), m);
        }
        string ini;
        const bool have = ReadAll(Join(dir, L"OptiScaler.ini"), ini);
        o.j.B("optiscaler_route", HostedRecord(m)).B("ini_found", have).S("proxy", m.proxy).S("proxy_reason", m.proxyReason);
        wstring ver;
        if (!m.proxy.empty() && IsFile(Join(dir, m.proxy))) ver = FileVersion(Join(dir, m.proxy));
        o.j.S("optiscaler_version", ver);
        o.j.Key("locked").Arr();
        const auto rows = OptiTabRows(ini);
        for (auto& r : rows) if (r.locked) o.j.Obj().S("section", r.section).S("key", r.key).S("value", r.value).S("why", r.why ? r.why : L"").End();
        o.j.End().Key("editable").Arr();
        for (auto& r : rows) {
            if (r.locked) continue;
            o.j.Obj().S("section", r.section).S("key", r.key).S("value", r.value).S("label", r.def->label).S("help", r.def->help);
            for (auto& c : m.inikeys) if (IEq(c.file, L"OptiScaler.ini") && c.section == r.section && c.key == r.key) o.j.S("original", c.original);
            o.j.End();
            Print(FromUtf8(r.section) + L"." + FromUtf8(r.key) + L" = " + FromUtf8(r.value) + L"  (" + r.def->label + L")");
        }
        o.j.End().Key("outside_edits").Arr();
        for (auto& x : JournalOutside(ini, m.inikeys, L"OptiScaler.ini")) o.j.Obj().S("section", x.section).S("key", x.key).S("file", x.fileValue).S("ours", x.ours).End();
        o.j.End().Key("journal").Arr();
        for (auto& c : m.inikeys) o.j.Obj().S("file", c.file).S("section", c.section).S("key", c.key).B("absent", c.absent).S("original", c.original).S("ours", c.ours).End();
        o.j.End();
        return EXIT_OK;
    }
    if (verb == L"clean-leftovers" || verb == L"adopt-leftovers") {
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        if (NeedsElevation(dir)) { o.j.S("error", L"administrator permission is needed for " + dir); return EXIT_REFUSED; }
        OpRequest r; ParseOp(verb, r.op); r.exe = exe; r.dir = dir;
        OpLogBegin();
        Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
        const OpResult res = RunOp(g.ctx, r, ui);
        OpLogEnd();
        o.j.S("operation", verb).S("dir", dir);
        ResultJson(o, res);
        return res.code;
    }
    if (verb == L"install" || verb == L"repair" || verb == L"update" || verb == L"uninstall" || verb == L"rollback" || verb == L"adopt") {
        Op op; ParseOp(verb, op);
        if (verb == L"adopt" && a.has(L"all")) {
            o.j.Key("adopted").Arr();
            int code = EXIT_OK;
            for (auto& e : ScanGames(g.ctx, scan, noScan)) {
                if (!e.hasRecord || !e.adoptable) continue;
                OpLogBegin();
                OpRequest r; r.op = Op::Adopt; r.dir = e.dir;
                Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
                const OpResult res = RunOp(g.ctx, r, ui);
                OpLogEnd();
                o.j.Obj().S("dir", e.dir).N("code", res.code).S("message", res.message).Key("ignored").Arr();
                for (auto& n : res.notes) if (n.rfind(kIgnoredPrefix, 0) == 0) o.j.Str(n.substr(wcslen(kIgnoredPrefix), n.find(L" (") - wcslen(kIgnoredPrefix)));
                o.j.End().End();
                Print(e.dir + L": " + res.message);
                for (auto& n : res.notes) if (n.rfind(kIgnoredPrefix, 0) == 0) Print(L"  " + n);
                if (res.code) code = res.code;
            }
            o.j.End();
            return code;
        }
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        OpRequest r; r.op = op; r.exe = exe; r.dir = dir;
        r.optiPackage = op == Op::Install || op == Op::Update || op == Op::Repair ? OptiFor(a, gs) : (!a.get(L"opti").empty() ? a.get(L"opti") : gs.optiPackage);
        r.modelDll = a.get(L"model");
        r.moveMods = a.has(L"switch") || a.has(L"movemods");
        r.removeOld = a.has(L"removeold") || a.has(L"clean-leftovers");
        r.experimental = a.has(L"experimental");
        r.force = a.has(L"force");
        r.noScan = noScan;
        r.hasProfile = true; r.profile = gs.defaults;
        OpLogBegin();
        Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
        OpResult res = RunOp(g.ctx, r, ui);
        // the RR data this install may have converted: the RR games installed before it follow (no prompt from here: the
        // ones in protected folders are said)
        if (!res.code && (op == Op::Install || op == Op::Repair || op == Op::Update)) SyncRrSettings(nullptr, r.dir, res.warnings);
        if (!res.code && (op == Op::Install || op == Op::Repair || op == Op::Update)) RrConfigAfterInstall(a, r.dir, res, o);
        OpLogEnd();
        o.j.S("operation", OpName(op)).S("dir", r.dir);
        ResultJson(o, res);
        return res.code;
    }
    if (verb == L"apply") {  // the elevated worker: one job (--job, --result), or several in one run (--batch: one prompt)
        bool hiveSet = false;
        auto runJob = [&hiveSet](const wstring& job, const wstring& result) {
            string text;
            OpRequest r; wstring lad, logFile, why, sid, docs;
            int perEntry = -1;
            OpResult res;
            if (job.empty() || result.empty() || !ReadAll(job, text)) { res.code = EXIT_USAGE; res.message = L"apply needs --job=<file> and --result=<file>"; }
            else if (!ParseJob(text, r, lad, logFile, &why, &sid, &perEntry, &docs)) { res.code = EXIT_USAGE; res.message = L"bad job file: " + why; }
            else {
                SetEnvironmentVariableW(L"LOCALAPPDATA", lad.c_str());  // the player's model and shared data, not this account's
                g.ctx.playerDocuments = docs;  // and the player's Documents (rrconfig.h), not this account's
                if (!logFile.empty()) g.ctx.logFile = logFile;
                // run by another account (an administrator's name typed at the prompt): the Apps & features entries go to the
                // player's hive, where the player's launcher and Settings > Apps look
                HKEY hive = nullptr;
                if (!hiveSet && !sid.empty() && !IEq(sid, CurrentUserSid())) {
                    hiveSet = true;
                    if (RegOpenKeyExW(HKEY_USERS, sid.c_str(), 0, KEY_READ | KEY_WRITE, &hive) == ERROR_SUCCESS) SetUninstallHive(hive);
                    else LogLine(g.ctx, L"elevated worker: the player's registry hive cannot be opened; Apps & features entries go to this account");
                }
                g.ctx.perGameEntry = perEntry >= 0 ? perEntry == 1 : !LauncherInstalled();
                Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
                LogLine(g.ctx, wstring(L"elevated worker: ") + OpName(r.op) + L" " + (r.dir.empty() ? r.exe : r.dir) + (IsElevated() ? L" (elevated)" : L" (not elevated)"));
                res = RunOp(g.ctx, r, ui);
            }
            if (!result.empty()) { const string rt = ResultText(res); try { WriteFileAtomic(result, rt.data(), rt.size()); } catch (const SetupError&) {} }
            return res;
        };
        if (!a.get(L"batch").empty()) {
            string text;
            std::vector<std::pair<wstring, wstring>> jobs;
            bool stop = false, failed = false;
            OpResult last;
            if (!ReadAll(a.get(L"batch"), text) || !ParseBatch(text, jobs, stop)) { last.code = EXIT_USAGE; last.message = L"apply --batch needs a file of job= and result= lines"; }
            int ran = 0;
            for (auto& j : jobs) {
                if (stop && failed) {  // an earlier game failed: this one is not run, and its result says so
                    OpResult skip; skip.code = kOpNotRun; skip.message = L"not run: an earlier game failed";
                    const string rt = ResultText(skip);
                    try { WriteFileAtomic(j.second, rt.data(), rt.size()); } catch (const SetupError&) {}
                    continue;
                }
                const OpResult res = runJob(j.first, j.second);
                ++ran;
                if (res.code) { failed = true; last = res; }
            }
            o.j.N("jobs", (long long)jobs.size()).N("ran", ran);
            ResultJson(o, last);
            return last.code;
        }
        const OpResult res = runJob(a.get(L"job"), a.get(L"result"));
        ResultJson(o, res);
        return res.code;
    }
    if (verb == L"set" || verb == L"get") {
        if (a.has(L"global")) {
            if (verb == L"set") {
                for (auto& kv : a.pos) {
                    const size_t eq = kv.find(L'=');
                    wstring why;
                    if (eq == wstring::npos || !SetGlobal(gs, ToUtf8(kv.substr(0, eq)), ToUtf8(kv.substr(eq + 1)), &why)) {
                        o.j.S("error", why.empty() ? L"use key=value" : why); Print(why); return EXIT_USAGE;
                    }
                }
                SaveGlobal(gl.paths, gs);
            }
            o.j.Key("global").Obj().S("optiscaler", gs.optiPackage).B("play_button", gs.playButton).B("update_check", gs.updateCheck).S("update_url", gs.updateUrl).S("appearance", gs.appearance).S("accent", gs.accent).B("exact_mode", gs.exactMode)
                .S("data_dir", gl.paths.dataDir).B("portable", gl.paths.portable).B("installed", LauncherInstalled());
            o.j.Key("extra_folders").Arr(); for (auto& f : gs.extraFolders) o.j.Str(f); o.j.End();
            o.j.Key("defaults").Obj(); for (size_t i = 0; i < kSettingCount; ++i) o.j.S(kSettings[i].key, gs.defaults.get(kSettings[i].key)); o.j.End();
            o.j.End();
            return EXIT_OK;
        }
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        Record m;
        if (!LoadRecord(RecordPath(dir), m) || m.legacy) { o.j.S("error", L"WrongGPU is not installed in " + dir); Print(L"WrongGPU is not installed in " + dir); return EXIT_NOT_INSTALLED; }
        Profile p = GameProfile(dir);
        std::vector<std::pair<string, string>> optiEdits;
        if (verb == L"set") {
            for (size_t i = 1; i < a.pos.size(); ++i) {
                const wstring& kv = a.pos[i];
                const size_t eq = kv.find(L'=');
                wstring why;
                if (eq == wstring::npos) why = L"use key=value: " + kv;
                else if (StartsWithI(kv, L"opti.")) {  // an OptiScaler.ini key (the OptiScaler tab): checked when applied
                    optiEdits.push_back({ToUtf8(kv.substr(5, eq - 5)), ToUtf8(kv.substr(eq + 1))});
                    continue;
                }
                if (why.empty()) {  // a game's experimental opt-in (optiscaler.h kOptiGameKeys): only for its game
                    const wchar_t* only = OptiGameSettingExe(ToUtf8(kv.substr(0, eq)).c_str());
                    if (only && !IEq(BaseName(m.exe), only))
                        why = kv.substr(0, eq) + L" is a setting of " + only + L" only";
                    if (IEq(kv.substr(0, eq), L"rr_config_fix") && !RrConfigGameFor(m.exe))  // rrconfig.h: those two games only
                        why = L"rr_config_fix is a setting of " + RrConfigGameExes() + L" only";
                }
                if (!why.empty() || !SetSetting(p, ToUtf8(kv.substr(0, eq)), ToUtf8(kv.substr(eq + 1)), &why)) {
                    o.j.S("error", why); Print(why); return EXIT_USAGE;
                }
            }
            if (NeedsElevation(dir)) { o.j.S("error", L"administrator permission is needed for " + dir); Print(L"administrator permission is needed for " + dir); return EXIT_REFUSED; }
            OpLogBegin();
            OpRequest r; r.op = Op::Apply; r.dir = dir; r.hasProfile = true; r.profile = p; r.optiEdits = optiEdits;
            Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
            const OpResult res = RunOp(g.ctx, r, ui);
            OpLogEnd();
            ResultJson(o, res);
            if (res.code) return res.code;
        }
        string wini, oini;
        ReadAll(Join(dir, L"wronggpu.ini"), wini);
        ReadAll(Join(dir, L"OptiScaler.ini"), oini);
        auto eff = EffectiveSettings(wini, oini);
        if (const RrFileState rs = ReadRrSettings(dir); rs.present) {  // Ray Reconstruction: the RR plugin's settings file
            eff["rr_enabled"] = rs.enabled ? "on" : "off";
            if (!rs.mode.empty()) eff["rr_mode"] = rs.mode;
        }
        // rr_config_fix (rrconfig.h): on while the record says the game's Engine.ini has WrongGPU's line
        eff["rr_config_fix"] = m.rrconfig.present ? "on" : p.get("rr_config_fix") == "on" ? "off" : p.get("rr_config_fix");
        if (const RrConfigGame* rg = RrConfigGameFor(m.exe))
            // waiting: chosen before the game made its settings folder (-1: added after its first start, 1: the folder is there
            // now and the next Apply adds it; r2b #1)
            o.j.Key("rr_config").Obj().S("game", rg->name).B("fixed", m.rrconfig.present).N("waiting", RrConfigWaiting(g.ctx, m, p.get("rr_config_fix")))
                .S("file", m.rrconfig.present ? m.rrconfig.path : RrConfigFile(g.ctx, *rg, nullptr)).End();
        o.j.Key("settings").Obj();
        for (size_t i = 0; i < kSettingCount; ++i) {
            const SettingDef& d = kSettings[i];
            if (verb == L"get" && a.pos.size() > 1 && !IEq(a.pos[1], FromUtf8(d.key))) continue;
            o.j.Key(d.key).Obj().S("profile", p.get(d.key)).S("file", eff.at(d.key)).S("default", d.def).End();
            if (verb == L"get") Print(FromUtf8(d.key) + L" = " + FromUtf8(p.get(d.key)) + (eff.at(d.key) != p.get(d.key) ? L"  (file: " + FromUtf8(eff.at(d.key)) + L")" : L""));
        }
        o.j.End();
        o.j.Key("outside_edits").Arr();
        for (auto& x : OutsideEdits(p, wini, oini)) o.j.Obj().S("key", x.key).S("file", x.fileValue).S("profile", x.profileValue).End();
        o.j.End();
        o.j.S("runtime_options_line", [&] { string v, s; return IniGetFrontEnd(wini, "runtime_options", &v, &s) ? v : string(); }());
        return EXIT_OK;
    }
    if (verb == L"collect") {
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        SystemInfo sys; sys.win = WindowsInfo(); sys.gpu = DetectGpu();
        try {
            const Bundle b = CollectLogs(g.ctx, dir, a.get(L"out-zip").empty() ? DefaultBundlePath(dir) : a.get(L"out-zip"), sys);
            o.j.S("zip", b.zip).Key("files").Arr(); for (auto& f : b.files) o.j.Str(f); o.j.End();
            Print(L"logs collected: " + b.zip);
        } catch (const SetupError& e) { o.j.S("error", e.msg); Print(e.msg); return e.code; }
        return EXIT_OK;
    }
    if (verb == L"hipcheck") {
        const HipInfo h = CheckHip(g.ctx);
        o.j.B("ok", h.ok).S("dll", h.dll).S("device", h.device).S("arch", h.arch).S("text", h.text)
            .S("note", L"The HIP runtime is optional: Model M runs on the D3D12 executor; HIP is needed for the HIP executor, sparkle guard and frame capture.");
        Print(h.ok ? L"HIP: " + h.device + L" (" + h.arch + L") with " + h.dll : L"HIP: " + h.text + L" (optional: the D3D12 executor needs no HIP)");
        return EXIT_OK;  // a warning only
    }
    if (verb == L"model") {
        const wstring sub = a.pos.empty() ? L"status" : a.pos[0];
        // --preset=<name>: the model by its name (M, L; K in a build with Preset K). Its converter map, model folder and
        // messages; without it, the launcher's model as before.
        if (a.opt.count(L"preset") || a.has(L"preset")) {  // --preset=<name>; a bare --preset (no name) is refused
            wstring why;
            if (!SelectPreset(g.ctx, a.get(L"preset"), why)) { o.j.S("error", why); Print(why); return EXIT_USAGE; }
        }
        o.j.S("preset", wstring(1, SelectedPreset(g.ctx)));
        if (sub == L"convert") {
            if (!ConvertIgnoreArgs(a, o)) return o.code;
            OpLogBegin();
            const std::vector<StoreGame> libs = ConvertLibraries(a, noScan);
            ModelState m = FindModel(g.ctx, nullptr, a.get(L"model").empty() ? wstring() : FullPath(a.get(L"model")), libs);
            IgnoreJson(o, m.ignored);
            int code = EXIT_OK;
            std::vector<wstring> notes = m.notes;  // the player's lines (a chosen file on a list, a model kept unchecked, ...)
            if (m.ok()) {
                Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
                try { const ModelResult r = WriteModel(g.ctx, m, ui); notes = r.notes; o.j.S("outcome", r.outcome); Print(L"model: " + r.outcome); }
                catch (const SetupError& e) { o.j.S("error", e.msg); Print(e.msg); code = e.code; }
            } else {
                o.j.S("error", m.error); Print(L"model: " + m.error);
                code = m.damaged ? EXIT_DAMAGED : m.code ? m.code : EXIT_NO_MODEL;  // 10: an ignore list can't be read
            }
            o.j.Key("notes").Arr();
            for (auto& n : notes) { o.j.Str(n); Print(L"  " + n); }
            o.j.End();
            OpLogEnd();
            if (code) return code;
        }
        wstring why;
        wgc::ModelFileInfo info{};
        const bool valid = ExistingModelValid(g.ctx, &why, &info);
        // map_sha256: the converter map of the selected model (embedded; the file it was built from has the same SHA-256)
        const wstring mapSha = g.ctx.map.map ? FromUtf8(wgc::to_hex(g.ctx.map.map->file_sha256)) : wstring();
        o.j.Key("model").Obj().S("file", ModelFile(g.ctx)).B("valid", valid).S("source", ReadModelSource(g.ctx)).S("why", why)
            .S("map_sha256", mapSha).S("map_error", g.ctx.mapError).End();
        Print(wstring(L"model data: ") + (valid ? L"ready (" + ModelFile(g.ctx) + L")" : L"not ready" + (why.empty() ? wstring() : L" (" + why + L")")));
        return valid || sub == L"status" ? EXIT_OK : EXIT_NO_MODEL;
    }
    if (verb == L"rr-prepare") {  // release 0.6.0: Ray Reconstruction's kernels prepared for an installed game (rrprep.h);
        // --passes=1..3 (default: the game's RR mode), --check: only whether they are prepared
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        if (exe.empty()) { Record m; if (LoadRecord(RecordPath(dir), m)) exe = m.exe; }
        const int passes = a.get(L"passes").empty() ? RrPassesFor(ReadRrSettings(dir).mode) : std::max(1, std::min(3, _wtoi(a.get(L"passes").c_str())));
        const wstring rt = IsFile(Join(Parent(exe), kRrRuntime)) ? Join(Parent(exe), kRrRuntime) : Join(dir, kRrRuntime);
        wstring why, notWhy;
        int rc = -1;
        unsigned kernels = 0;
        const ULONGLONG t0 = GetTickCount64();
        if (!a.has(L"check")) {
            OpLogBegin();
            rc = PrepareRr(g.ctx, gl.paths.dataDir, exe, rt, passes, [&kernels](const RrPrepProgress& p) { kernels = p.done; return true; }, &why);
            OpLogEnd();
        }
        const RrPrepState state = RrPrepStateOf(g.ctx, gl.paths.dataDir, exe, passes, &notWhy);
        const bool ready = state == RrPrepState::Ready;
        o.j.Key("rr_prepare").Obj().S("exe", exe).S("runtime", rt).N("passes", passes).N("result", rc).N("kernels", kernels)
            .N("ms", (long long)(GetTickCount64() - t0)).S("why", why).B("ready", ready).S("not_ready_why", notWhy).S("state", RrPrepStateName(state))
            .S("driver", RrDriverVersion()).End();
        Print(wstring(L"Ray Reconstruction kernels for ") + BaseName(exe) + L": " + (ready ? L"prepared" : L"not prepared (" + notWhy + L")") +
              (why.empty() ? L"" : L"; " + why));
        // 4: the game prepares them at its own first start (gamerules.h rrKernelsInGame): nothing to do here
        return rc <= 0 || rc == 2 || rc == 4 ? (ready || rc == 2 || rc == 4 || a.has(L"check") ? EXIT_OK : EXIT_FAILED) : EXIT_FAILED;
    }
    if (verb == L"rr-recheck") {  // release 0.6.0 polish: what the window's start does for a game (rrprep.h RrStaleAfterUpdate):
        // prepared before, and the graphics driver or WrongGPU's runtime changed since -> prepared again (default mode first,
        // then the other passes its mode needs); --run does it, else it only says whether the start would
        wstring exe, dir;
        if (!Game(a, o, exe, dir)) return o.code;
        if (exe.empty()) { Record m; if (LoadRecord(RecordPath(dir), m)) exe = m.exe; }
        const int passes = RrPassesFor(ReadRrSettings(dir).mode);
        const wstring rt = IsFile(Join(Parent(exe), kRrRuntime)) ? Join(Parent(exe), kRrRuntime) : Join(dir, kRrRuntime);
        wstring why;
        // the start looks at the default mode first: a game whose default-mode kernels are fine is not stale
        const RrPrepState before = RrPrepStateOf(g.ctx, gl.paths.dataDir, exe, 1, &why);
        const bool stale = RrStaleAfterUpdate(before);
        int rc = -1, rc2 = -1;
        wstring runWhy;
        if (stale && a.has(L"run")) {
            OpLogBegin();
            LogLine(g.ctx, L"Ray Reconstruction: " + BaseName(exe) + L" prepared again (" + why + L")");
            rc = PrepareRr(g.ctx, gl.paths.dataDir, exe, rt, 1, [](const RrPrepProgress&) { return true; }, &runWhy);
            if (rc == 0 && passes > 1) rc2 = PrepareRr(g.ctx, gl.paths.dataDir, exe, rt, passes, [](const RrPrepProgress&) { return true; }, &runWhy);
            OpLogEnd();
        }
        wstring nowWhy;
        const RrPrepState now = RrPrepStateOf(g.ctx, gl.paths.dataDir, exe, passes, &nowWhy);
        o.j.Key("rr_recheck").Obj().S("exe", exe).B("stale", stale).S("state_before", RrPrepStateName(before)).S("why", why)
            .N("result", rc).N("result_more_passes", rc2).S("run_why", runWhy).S("state", RrPrepStateName(now)).S("driver", RrDriverVersion()).End();
        Print(wstring(L"Ray Reconstruction kernels for ") + BaseName(exe) + L": " + RrPrepStateName(before) +
              (stale ? (a.has(L"run") ? L" -> prepared again: " + wstring(RrPrepStateName(now)) : L" (the launcher's start prepares them again)") : L""));
        return stale && a.has(L"run") && rc != 0 && rc != 2 ? EXIT_FAILED : EXIT_OK;
    }
    if (verb == L"rr-data") {  // Ray Reconstruction data (rr.h): status, or "convert" [--dll=<RR model file>]
        const wstring sub = a.pos.empty() ? L"status" : a.pos[0];
        if (sub == L"convert") {
            if (!ConvertIgnoreArgs(a, o)) return o.code;
            OpLogBegin();
            const std::vector<StoreGame> libs = ConvertLibraries(a, noScan);
            wstring why;
            std::vector<RrSettingsFailure> notWritten;
            std::vector<wstring> warnings;
            IgnoreReport ignored;
            const bool ok = ConvertRrData(g.ctx, a.get(L"dll").empty() ? wstring() : FullPath(a.get(L"dll")), libs, &why, &notWritten, &ignored);
            if (ok) SyncRrSettings(nullptr, L"", warnings, &notWritten);
            OpLogEnd();
            IgnoreJson(o, ignored);
            if (!ok) { o.j.S("error", why); Print(L"Ray Reconstruction data: " + why); return ignored.unreadable.empty() ? EXIT_NO_MODEL : EXIT_USAGE; }
            o.j.Key("warnings").Arr(); for (auto& w : warnings) { o.j.Str(w); Print(L"  warning: " + w); } o.j.End();
        }
#ifdef WRONGGPU_TESTING
        // tests: what an install does with the RR data (FindRrData without `fresh`: the data on this PC reused when the ignore
        // lists allow it, else converted again from --library= games), without a game or a device
        if (sub == L"prepare") {
            if (!ConvertIgnoreArgs(a, o)) return o.code;
            const RrDataState st = FindRrData(g.ctx, nullptr, L"", ConvertLibraries(a, noScan), false);
            wstring why;
            const bool ok = st.ok && WriteRrData(g.ctx, st, &why);
            IgnoreJson(o, st.ignored);
            o.j.B("reused", st.reused).Key("notes").Arr();
            for (auto& n : st.notes) { o.j.Str(n); Print(L"  " + n); }
            o.j.End();
            if (!ok) {
                o.j.S("error", why.empty() ? st.error : why); Print(L"Ray Reconstruction data: " + (why.empty() ? st.error : why));
                return st.ignored.unreadable.empty() ? EXIT_NO_MODEL : EXIT_USAGE;
            }
        }
#endif
        wstring why;
        const bool valid = RrDataValid(g.ctx, &why);
        o.j.Key("rr_data").Obj().S("file", RrDataFile()).B("valid", valid).S("source", RrDataSource()).S("why", why).B("payload", RrPayloadAvailable(g.ctx)).End();
        Print(wstring(L"Ray Reconstruction data: ") + (valid ? L"ready (" + RrDataFile() + L")" : L"not ready" + (why.empty() ? wstring() : L" (" + why + L")")));
        return valid || sub == L"status" ? EXIT_OK : EXIT_NO_MODEL;
    }
    if (verb == L"self-install") {
        wstring why;
        std::vector<wstring> kept;
        if (!InstallLauncherForUser(a.get(L"dir"), a.get(L"startmenu"), nullptr, gl.global, {}, why, &kept)) { o.j.S("error", why); Print(why); return EXIT_FAILED; }
        o.j.Key("kept_own_entry").Arr(); for (auto& k : kept) o.j.Str(k); o.j.End();
        Print(L"WrongGPU is installed for this user.");
        return EXIT_OK;
    }
    if (verb == L"self-uninstall") {  // no window: the games are done in this process (Apps & features opens the window: uninstall-launcher)
        // nothing installed for this user: nothing to uninstall (never the games, nor this copy's own settings and logs), as the
        // window's Uninstall launcher is not offered then
        if (!LauncherInstalled()) {
            const wstring why = L"WrongGPU (the launcher) is not installed for this user: nothing was changed.";
            o.j.S("error", why); Print(why); return EXIT_NOT_INSTALLED;
        }
        Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); Print(s); };
        wstring why;
        std::vector<wstring> unreachable;
        if (!UninstallLauncher(!a.has(L"keep-games"), a.get(L"startmenu"), nullptr, ui, why, &unreachable)) { o.j.S("error", why); Print(why); return EXIT_FAILED; }
        o.j.Key("unreachable").Arr(); for (auto& d : unreachable) o.j.Str(d); o.j.End();
        for (auto& d : unreachable) Print(L"left as it is (its drive is not connected): " + d);
        Print(L"WrongGPU (the launcher) is uninstalled.");
#ifdef WRONGGPU_TESTING
        // tests: a Repair in this same process after the uninstall, as a portable window that stays open runs it (the game's
        // own Apps & features entry follows the launcher's state now, not the state this process started with)
        if (!a.get(L"repair-after").empty()) {
            wstring exe, dir;
            if (!ResolveGameArg(g.ctx, a.get(L"repair-after"), exe, dir)) { o.j.S("error", L"no game found at " + a.get(L"repair-after")); return EXIT_REFUSED; }
            OpRequest r; r.op = Op::Repair; r.exe = exe; r.dir = dir;
            r.optiPackage = OptiFor(a, gs);
            r.experimental = a.has(L"experimental"); r.force = a.has(L"force"); r.noScan = noScan;
            r.hasProfile = true; r.profile = gs.defaults;
            const OpResult res = RunOp(g.ctx, r, ui);
            o.j.N("repair_code", res.code).S("repair_message", res.message);
            if (res.code) return res.code;
        }
#endif
        return EXIT_OK;
    }
    if (verb == L"remove-data") {  // a portable copy's Settings > Remove data... without the window (RemoveSharedData)
        const SharedDataLook look = LookAtSharedData(g.ctx);
        wstring why, kept;
        if (!RemoveSharedData(g.ctx, &why, &kept)) { o.j.S("error", why); Print(why); return EXIT_REFUSED; }
        o.j.B("removed", look.there).S("kept", kept);
        Print(look.there ? L"WrongGPU's data is removed from this PC." : L"There is no WrongGPU data to remove.");
        if (!kept.empty()) Print(kept);  // the ignore list stays (convert_ignore.h)
        return EXIT_OK;
    }
    if (verb == L"play") {
        wstring exe, dir, why;
        if (!Game(a, o, exe, dir)) return o.code;
        PlayWay how;
        const bool ok = PlayGame(exe, &why, &how, &g.ctx);
        // which way it started: "steam" (steam://rungameid/<appid>), "epic" (com.epicgames.launcher://apps/<epic_app>, rc6) or
        // "direct" (the exe); steam_note / epic_note: why a Steam / Epic game started directly; appid: the game's Steam app id,
        // epic_app: its Epic AppName (empty: none)
        o.j.S("appid", how.appid);
        o.j.S("epic_app", how.epicApp);
        if (!how.note.empty()) o.j.S(how.appid.empty() && !how.epicApp.empty() ? "epic_note" : "steam_note", how.note);
        if (!ok) { o.j.S("error", why); Print(why); return EXIT_REFUSED; }
        o.j.S("started", how.steam ? L"steam" : how.epic ? L"epic" : L"direct");
        Print(how.steam  ? L"Started " + BaseName(exe) + L" through Steam (app id " + how.appid + L")."
              : how.epic ? L"Started " + BaseName(exe) + L" through Epic (app " + how.epicApp + L")."
                         : L"Started " + BaseName(exe) + L" directly" + (how.note.empty() ? wstring() : L" (" + how.note + L")") + L".");
        return EXIT_OK;
    }
    if (verb == L"ui-dump") {  // development and tests: the window built hidden, its pages as JSON (no dialog, no change)
        gs.extraFolders = scan.extraFolders;
        const wstring dump = a.get(L"dump");
        if (dump.empty()) { o.j.S("error", L"ui-dump needs --dump=<file.json>"); return EXIT_USAGE; }
        const int code = RunLauncherUiDump(FullPath(dump), a.get(L"shots").empty() ? wstring() : FullPath(a.get(L"shots")), noScan, a.get(L"apply"));
        o.j.S("dump", FullPath(dump));
        return code;
    }
    if (verb == L"ui-render") {  // development: fixture screenshots of the window (nothing shown, no game folder read)
        const wstring out = a.get(L"out-dir"), fixture = a.get(L"fixture");
        if (out.empty() || fixture.empty()) { o.j.S("error", L"ui-render needs --out-dir=<folder> --fixture=<file>"); return EXIT_USAGE; }
        int w = 1180, h = 760;
        if (!a.get(L"size").empty()) swscanf(a.get(L"size").c_str(), L"%dx%d", &w, &h);
        std::vector<float> scales;
        wstring list = a.get(L"scales").empty() ? wstring(L"1,1.5") : a.get(L"scales");
        for (size_t p = 0; p < list.size();) {
            const size_t c = list.find(L',', p);
            scales.push_back((float)_wtof(list.substr(p, c == wstring::npos ? wstring::npos : c - p).c_str()));
            p = c == wstring::npos ? list.size() : c + 1;
        }
        wstring why;
        const int code = RunLauncherUiRender(FullPath(out), FullPath(fixture), std::max(320, w), std::max(240, h), scales, a.get(L"frames"), Lower(a.get(L"theme")), &why);
        if (code) o.j.S("error", why);
        o.j.S("out_dir", FullPath(out));
        return code;
    }
    if (verb == L"update-check") {
        const UpdateInfo u = CheckForUpdate(gs, g.ctx.data->version);
        o.j.B("checked", u.checked).B("newer", u.newer).S("version", u.version).S("page", u.page).S("message", u.message);
        Print(u.message);
        return EXIT_OK;
    }
    o.j.S("error", L"unknown command " + verb);
    Print(L"unknown command " + verb + L" (WrongGPU.exe version | scan | status | install | repair | update | uninstall | rollback | adopt | clean-leftovers | adopt-leftovers | "
          L"set | get | opti | coexist | fingerprint | collect | hipcheck | model | self-install | self-uninstall | remove-data | play | update-check)");
    return EXIT_USAGE;
}

}  // namespace

int RunCli(int argc, wchar_t** argv) {
    const wstring verb = Lower(argv[1]);
    Args a;
    wstring logFile;
    for (int i = 2; i < argc; ++i) {
        const wstring s = argv[i];
        if (s.rfind(L"--", 0) == 0) {
            const size_t eq = s.find(L'=');
            const wstring name = Lower(s.substr(2, eq == wstring::npos ? wstring::npos : eq - 2));
            auto put = [&a, &name](const wstring& v) {
                if (name == L"library") a.libraries.push_back(v);
                else if (name == L"ignore") a.ignores.push_back(v);
                else if (name == L"ignore-list") a.ignoreLists.push_back(v);
                else a.opt[name] = v;
            };
            if (eq != wstring::npos) {
                put(s.substr(eq + 1));
            } else if ((name == L"out" || name == L"job" || name == L"result" || name == L"batch" || name == L"data" || name == L"log" || name == L"dir" || name == L"opti" ||
                        name == L"model" || name == L"library" || name == L"startmenu") && i + 1 < argc) {  // (--ignore and --ignore-list need their "=")
                put(argv[++i]);
            } else a.flags.insert(name);
        } else a.pos.push_back(s);
    }
    gl.dataOverride = a.get(L"data");
    logFile = a.get(L"log");
    LauncherInit(logFile, verb == L"apply");
    Output o;
    o.outFile = a.get(L"out");
    // `collect` writes a zip: --out is the JSON as everywhere; --zip=<file> names the zip
    if (!a.get(L"zip").empty()) a.opt[L"out-zip"] = a.get(L"zip");
    o.j.Obj().S("verb", verb).S("version", g.ctx.data->version);
    LogLine(g.ctx, L"command: " + verb);
    int code = EXIT_FAILED;
    try { code = Verb(verb, a, o); }
    catch (const SetupError& e) { code = e.code; o.j.S("error", e.msg); Print(e.msg); }
    o.code = code;
    return o.finish();
}

}  // namespace wg::app
