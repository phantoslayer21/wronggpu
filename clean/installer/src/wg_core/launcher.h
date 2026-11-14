// SPDX-License-Identifier: MIT
// WrongGPU launcher core: what the launcher does on top of the installer's engine (build/launcher-design.md sections
// 2, 3 and 6, milestone M1): where its data lives (installed per user or portable), its global settings, the game
// list with a status per game, the per-game operations (install, repair, update, uninstall, roll back, apply
// settings, adopt a 0.5.x install), the job file of the elevated worker, its own install with one Apps & features
// entry, starting a game (Play) and the opt-in update check. No UI here: the GUI and the command line
// (src/app) call these. Written for this project.
#pragma once
#include "checks.h"
#include "install.h"
#include "leftovers.h"
#include "logtail.h"
#include "profile.h"

namespace wg {

// ---- data locations --------------------------------------------------------------------------------------------
// Installed: the launcher runs from %LOCALAPPDATA%\Programs\WrongGPU and keeps its data in %LOCALAPPDATA%\WrongGPU.
// Portable: a file "WrongGPU.portable" next to the exe; the data lives in <exe folder>\WrongGPU-data. The model data
// is always in %LOCALAPPDATA%\WrongGPU\model (the runtime reads it there), and every game keeps its record, profile
// and backup next to the game.
const wchar_t* const kPortableMarker = L"WrongGPU.portable";
const wchar_t* const kLauncherEntry = L"WrongGPU";  // the one Apps & features entry (UninstallRoot())
struct LauncherPaths {
    wstring exe, dataDir;
    bool portable = false;     // portable mode (marker next to the exe)
    bool installedCopy = false;  // running from the per-user install folder
    bool custom = false;       // --data= given
};
wstring DefaultInstallDir();   // %LOCALAPPDATA%\Programs\WrongGPU
LauncherPaths ResolvePaths(const wstring& selfExe, const wstring& dataOverride);
bool LauncherInstalled(wstring* exe = nullptr);  // the Apps & features entry exists and so does its exe
// A launcher started next to an installed copy (not the installed copy itself): what it does (rc5). Versions first; at the
// same version the build decides, never the version string alone: the same file (size and SHA-256; the installed copy is a
// verified byte copy) opens the installed copy, a different file is ordered by its link time (FileLinkTime). This copy
// offers to replace an older version or an older build; a newer version or a newer build opens the installed copy; an
// installed copy that isn't a readable program (no PE header: corrupt, cut short) never gets the start: this copy runs
// (RunHere). A newer running copy never defers to an older installed one.
enum class LauncherStart { OpenInstalled, OfferReplace, RunHere };
struct LauncherStartDecision {
    LauncherStart choice = LauncherStart::OpenInstalled;
    bool sameVersion = false, sameBuild = false;
    uint32_t selfBuilt = 0, installedBuilt = 0;  // link times (FileLinkTime)
    wstring why;                                 // for the log
};
LauncherStartDecision DecideLauncherStart(const wstring& selfExe, const wstring& selfVersion, const wstring& installedExe,
                                          const wstring& installedVersion);
wstring BuildTimeText(uint32_t linkTime);       // "2026-11-05 14:25 UTC"; "unknown" for 0
wstring CurrentUserSid();  // this process's account (the elevated worker's job names the player's)
// Started inside an app package (MSIX): its %LOCALAPPDATA% writes go to the package's private storage, where games do
// not see them. The window shows a warning.
bool RunningPackaged();

// ---- global settings (launcher.ini) ----------------------------------------------------------------------------
struct GlobalSettings {
    wstring optiPackage;           // the player's OptiScaler download (archive or folder) used for installs
    std::vector<wstring> extraFolders;  // folders whose sub-folders are games (in addition to Steam, Epic, GOG)
    std::vector<wstring> addedGames;    // game exes added by hand
    bool playButton = true;        // show a Play button on each game's page
    bool updateCheck = false;      // opt-in: look for a new release at start (off by default)
    wstring updateUrl;             // the release URL to ask (no default: given by the owner)
    wstring appearance = L"dark";  // the window's theme: dark (the default, owner 2026-11-01), light, or system (Windows' app mode)
    wstring accent = L"green";     // the controls' colour: green (the default), blue, purple, pink, cyan, graphite, windows
    // Settings > Model and runtime > Exact mode (off by default): a game's Ray Reconstruction quality offers Exact too.
    // Turning it off changes no game's setting (a game set to Exact keeps showing it).
    bool exactMode = false;
    bool firstRunDone = false;
    bool rrFirstStartNoted = false;  // the player was told once that the first RR start in a game can take a few minutes
    wstring window;                // the window's last place: "left,top,right,bottom,dpi,maximized" (screen pixels; empty: centred)
    Profile defaults;              // the profile a new install starts with
};
GlobalSettings LoadGlobal(const LauncherPaths& p);
void SaveGlobal(const LauncherPaths& p, const GlobalSettings& s);
// `set --global key=value` and the settings page: optiscaler, play_button, update_check, update_url, appearance, accent, exact_mode, extra_folder
// (adds), defaults.<profile key>.
bool SetGlobal(GlobalSettings& s, const string& key, const string& value, wstring* why);

// ---- game list -----------------------------------------------------------------------------------------------------
enum class GameStatus {
    Eligible, Installed, UpdateAvailable, NeedsRepair, TurnedOff, PlayersOptiScaler, ForeignMod, OldInstall,
    Orphaned, Refused, Running, Leftovers,
    NeedsPrereq  // the game needs a tool of the player's (REFramework for RE Requiem, gamerules.h) that is not there: not installed
                 // (Install waits for it), or installed and the tool gone since (WrongGPU can't run there; hasRecord)
};
const wchar_t* StatusName(GameStatus s);   // "installed", "update-available", ... (status --json)
struct GameEntry {
    wstring name, exe, dir, source;
    wstring appid;                // Steam app id (appmanifest), empty for other stores (the window's art)
    GameStatus status = GameStatus::Eligible;
    wstring statusText;           // for the list ("Installed 0.5.5", "Refused: anti-cheat (BattlEye)", ...)
    wstring installedVersion;     // the record's version
    bool otherBuild = false;      // rc5: the same version, but the record's payload line isn't this launcher's (PayloadDiffers)
    bool hasRecord = false, recordV1 = false, perGameEntry = false, hosted = false, adoptable = false;
    bool repairSuggested = false;
    bool exeFound = false;        // the game's exe was there when the entry was read (the window's Play: no file check per frame)
    bool hasPrevious = false;     // a version Roll back goes to is kept (previous\), read with the entry (no file read per frame)
    wstring previous;             // its version
    bool wgOff = false;           // its profile turns WrongGPU off (wronggpu=off, AMD FSR); also while it runs (status Running)
    std::vector<wstring> problems;  // VerifyInstall, missing wronggpu.ini, ...
    // rc6 L2: installed next to a launcher stub or a root starter (an earlier build's choice, MisplacedFor): the game's own exe.
    // A problem then (Needs repair); Repair, Update or Install moves the install there (uninstalled here, installed at the game)
    wstring realExe;
    std::vector<wstring> found;     // other tools / mods found (the findings' "what", managed ones left out)
    std::vector<Finding> coexist;   // the rules table for this game (design section 5)
    ProxyChoice proxyPlan;          // where OptiScaler would go (not installed yet) or stays
    std::vector<Leftover> leftovers;  // WrongGPU files without a record (status Leftovers)
    LastRun last;
    wstring proxy;                 // OptiScaler's proxy DLL of our install
    // Ray Reconstruction (rr.h): only games that use it show anything of it
    bool rrUses = false;           // the game ships DLSS Ray Reconstruction (or is known to call it), or its install has the RR files
    wstring rrWhy;                 // what shows it (the file found, "known game", "installed")
    bool rrDirect = false;         // the game creates RR itself, not through the game interface (rr.h RrUse::direct): WrongGPU's
                                   // RR runs there only with WrongGPU's edition of OptiScaler
    bool optiEdition = false;      // the install's OptiScaler is WrongGPU's edition (rr.h RecordHasEdition; installed games only)
    bool rrInstalled = false;      // the install placed WrongGPU's RR files
    bool rrEnabled = true;         // WrongGPU\wronggpu.ini rr_enabled (an RR install)
    bool rrHeldOff = false;        // rc5: its RR waits for the game's experimental opt-in, which is off (gamerules.h): no RR
                                   // prepare, toast or question; the page says so
    bool rrNotOffered = false;     // rc5: the game's menu has no RR option (gamerules.h): rrUses false, the page's line instead
    wstring rrMode;                // its rr_mode as the game has it now (OptiScaler's menu saves it there): exact, accurate, fast
    bool rrPathTracingOnly = false;  // the game's RR comes only with its path tracing: RR off = no path tracing (gamerules.h)
    // Borderlands 4 / Hogwarts Legacy (rrconfig.h): rr_config_fix=on chosen before the game made its settings folder, nothing
    // applied then, and the folder is there now: an Apply adds the line (the window does it once it opens, r2b #1)
    bool rrConfigDue = false;
    // ...and the game's folder needs administrator permission (a write test, made while no line is applied yet: r3 #4): the window
    // never applies it by itself then (that would raise a permission prompt nobody clicked for); its card says "Ready to add"
    // with Fix it now, and Apply, Repair or Update add it too (r2c)
    bool rrConfigProtected = false;
    // a tool of the player's the game needs before WrongGPU can work in it (gamerules.h: REFramework for RE Requiem)
    wstring needs;                 // its name; empty: none
    bool needsFound = false;       // it is there (needs, the evidence in needsWhy)
    wstring needsWhy;              // what shows it ("dinput8.dll, folder reframework")
    wstring needsPage;             // its public page (Learn more)
    wstring needsGame;             // the game's short name in sentences ("RE Requiem")
};
// rc5: this launcher would update the game: an older version, or the same version with another build's files (otherBuild).
// One rule for the status, the Repair-and-update words and the Library (never the version string alone).
bool UpdateDue(const GameEntry& e, const Context& ctx);
// The installed version as the window shows it before an arrow: "0.5.5", or "0.6.0 (another build)".
wstring InstalledLabel(const GameEntry& e);
// The status of one game (inspects the folder; reads the record, the log tail and the files). quick: the installed files
// only checked for being there (not hashed) and no Ray Reconstruction folder walk (ScanGames' first look at the managed
// games, checked in full right after).
// rc6 L2: a launcher stub or a root starter without an install next to it is read as the game it leads to (MisplacedFor).
GameEntry EntryFor(Context& ctx, const wstring& exe, const wstring& name, const wstring& source, bool quick = false);
// rc6 L2: the game an install's exe stands for when that exe is a launcher stub (REDprelauncher.exe: bin\x64\Cyberpunk2077.exe)
// or a starter with the game below it (ResolveGameExe: Hogwarts Legacy's root exe, an Unreal starter); empty: exe is the game.
// An install next to such an exe is in the wrong place, and nothing is ever installed next to it.
wstring MisplacedFor(const wstring& exe);
// How that problem starts in GameEntry::problems ("WrongGPU is next to REDprelauncher.exe, the game's launcher, not the game's
// program bin\x64\Cyberpunk2077.exe: Repair moves it there"); the window words it apart from missing or changed files.
const wchar_t* const kMisplacedProblem = L"WrongGPU is next to ";
inline bool IsMisplacedProblem(const wstring& p) { return p.rfind(kMisplacedProblem, 0) == 0; }
// Every game the launcher knows: the store libraries (unless noScan), the extra folders, the games added by hand
// and the managed folders (records). Sorted: managed first, then by name. `managedFirst`, when given, gets the managed
// games (sorted the same way) as soon as they are read, before the store libraries are scanned (the launcher shows them
// while the rest is read); not called when there are none.
std::vector<GameEntry> ScanGames(Context& ctx, const GlobalSettings& gs, bool noScan,
                                 const std::function<void(const std::vector<GameEntry>&)>& managedFirst = {});
// A game named on the command line: an exe, a game folder or an install folder with a record.
// exe and dir: the game exe (empty for an orphaned install) and its install folder. False: no game there.
bool ResolveGameArg(Context& ctx, const wstring& arg, wstring& exe, wstring& dir);

// ---- operations ----------------------------------------------------------------------------------------------------
// Detach: the launcher is uninstalled and the game keeps WrongGPU: its own Apps & features entry again (record regkey).
enum class Op { Install, Repair, Update, Uninstall, Rollback, Apply, Adopt, CleanLeftovers, AdoptLeftovers, Detach };
const wchar_t* OpName(Op o);
bool ParseOp(const wstring& s, Op& o);
struct OpRequest {
    Op op = Op::Install;
    wstring exe;                   // the game (install, repair, update) or empty with dir
    wstring dir;                   // the install folder (uninstall, rollback, apply, adopt)
    wstring optiPackage;           // install/update/repair: the player's OptiScaler download
    wstring modelDll;              // a DLSS model file to try first
    bool moveMods = false, removeOld = false, experimental = false, force = false, noScan = true;
    bool hasProfile = false;       // apply: this profile (else the game's profile.ini); install: the profile a new install starts with
    Profile profile;
    // apply, the OptiScaler tab: edits of editable keys ("Section.Key", value); reset: the editable keys back to what the
    // install wrote (inikey journal, byte for byte); keep: values OptiScaler's menu saved over the launcher's are kept.
    std::vector<std::pair<string, string>> optiEdits;
    bool optiReset = false, optiKeep = false;
    // apply: the OptiScaler tab only (its reset, keep or edits): wronggpu.ini, the profile's keys and profile.ini stay as they
    // are (an outside change to them keeps its own Keep / Use buttons)
    bool optiOnly = false;
    // the window's own apply (the Engine.ini choice waiting for the game's settings folder, r2c): never asks Windows for
    // permission. A folder that needs it: nothing is run (StartOp; the player's click on Fix it now or Apply does it)
    bool noPrompt = false;
    // apply: the game's Engine.ini alone (Borderlands 4 / Hogwarts Legacy, rrconfig.h), as rr_config_fix is in the saved
    // profile.ini: Engine.ini, its backup and the install record's rrconfig line, nothing else (no other settings file, no
    // profile value). The Engine.ini choice applied once the game's settings folder is there, and Fix it now / Fix it again
    // (UI review r3 #1: an outside change to wronggpu.ini or OptiScaler.ini is never put back by it)
    bool rrConfigOnly = false;
    // ...with this rr_config_fix ("on" / "off") saved first, the profile's one value written into profile.ini as it is now (read
    // again by the operation, never a snapshot of it): the Engine.ini question's answer (UI review r4 #4: a full Apply of the
    // profile captured when it was asked could put back a settings change made since)
    string rrConfigSet;
};
struct OpResult { int code = EXIT_OK; wstring message; std::vector<wstring> warnings, notes; };
// Runs one operation in this process (the caller decides about elevation: NeedsElevation).
OpResult RunOp(Context& ctx, const OpRequest& r, Ui& ui);
bool NeedsElevation(const wstring& dir);
// Writes the profile into the game's files (wronggpu.ini, OptiScaler.ini), the record's hashes of those files and
// owned-files.json; saves profile.ini.
// warnings: what could not be done that the player should hear (rr_config_fix: the game's Engine.ini, rrconfig.h).
void ApplyProfile(Context& ctx, const wstring& dir, const Profile& p, std::vector<wstring>& notes, std::vector<wstring>* warnings = nullptr);
Profile GameProfile(const wstring& dir);  // profile.ini, else the profile its wronggpu.ini implies
// The OptiScaler tab: reset / keep / edits of OpRequest written into the installed OptiScaler.ini with the inikey
// journal in the record. Nothing to do: no change.
void ApplyOptiTab(Context& ctx, const wstring& dir, const OpRequest& r, std::vector<wstring>& notes);
// OpRequest::rrConfigOnly: the game's Engine.ini as the saved profile's rr_config_fix says (on: ApplyRrConfig; anything else:
// nothing; off is taken out by an Apply of the profile), its backup and the record's rrconfig line. Nothing else is written.
// set ("on" / "off", r4 #4): that rr_config_fix saved into profile.ini first (its one value; nothing else written there), then
// Engine.ini follows it (off: WrongGPU's own line, if the record has it, taken out).
void ApplyRrConfigOnly(Context& ctx, const wstring& dir, std::vector<wstring>& notes, std::vector<wstring>* warnings = nullptr, const string& set = "");
// 0.5.x install -> launcher: record v2, profile.ini from wronggpu.ini, no per-game Apps & features entry, the
// restorer as uninstall.exe, owned-files.json, on the managed list. Returns notes; *repair: repair suggested.
std::vector<wstring> AdoptInstall(Context& ctx, const wstring& dir, bool* repair = nullptr);
bool HasPrevious(const wstring& dir, wstring* version = nullptr);
// HasPrevious and the kept files fit the install's current layout (Roll back can put them back)
bool PreviousFits(const wstring& dir, const Record& m, wstring* version = nullptr);

// ---- the elevated worker's job file -------------------------------------------------------------------------------
// UTF-8 key=value lines: op, exe, dir, opti, model, flags, localappdata (the player's, so the model and the shared data
// are the player's even when an administrator account runs the worker), log, profile.<key>; usersid (the player's
// account: its Apps & features entries, when another account runs the worker) and perentry (1: a new install gets its own
// entry, 0: the player's launcher is installed), each left out when empty / -1.
// documents: the player's Documents folder (rrconfig.h; the worker may run as another account, whose Documents it is not).
string JobText(const OpRequest& r, const wstring& localAppData, const wstring& logFile, const wstring& userSid = L"", int perGameEntry = -1,
               const wstring& documents = L"");
bool ParseJob(const string& text, OpRequest& r, wstring& localAppData, wstring& logFile, wstring* why, wstring* userSid = nullptr, int* perGameEntry = nullptr,
              wstring* documents = nullptr);
string ResultText(const OpResult& res);
bool ParseResult(const string& text, OpResult& res);
// Several jobs for one elevated worker ("apply --batch=<file>", one permission prompt): job=<file> / result=<file> pairs in
// the order they run; stop=1: none runs after the first that fails (its result then has the code kOpNotRun).
string BatchText(const std::vector<std::pair<wstring, wstring>>& jobs, bool stop);
bool ParseBatch(const string& text, std::vector<std::pair<wstring, wstring>>& jobs, bool& stop);

// ---- the launcher's own install ------------------------------------------------------------------------------------
// Copies the exe to `dir` (default DefaultInstallDir()), adds the Start-menu shortcut (startMenu: a folder; empty:
// the user's Start menu, or where an installed copy's entry has its shortcut (a replace); "-": none; test builds: a folder
// in %LOCALAPPDATA%, never the player's Start menu) and the one Apps & features entry (its Uninstall: "WrongGPU.exe
// uninstall-launcher", the window's uninstall). A newer copy replaces the installed one through a checked copy and a
// rename (the old one may still run). False with *why.
bool SelfInstall(Context& ctx, const wstring& selfExe, const wstring& dir, const wstring& startMenu, wstring* why);
// The installed copy at start: its entry written again (version, size, the uninstall command) and an earlier replace's
// WrongGPU.exe.old deleted. Nothing when selfExe is not the installed copy.
void RefreshLauncherEntry(Context& ctx, const wstring& selfExe);
// The per-game operations of SelfUninstall, handed over all at once and in order: the window runs those whose folders need
// administrator rights in one elevated worker (one permission prompt). A result per request, in order; nothing runs after
// the first failure, and a request not run has the code kOpNotRun. Empty: each in this process.
const int kOpNotRun = -1;
using GameOps = std::function<std::vector<OpResult>(const std::vector<OpRequest>&)>;
// The games first, before anything of the launcher changes: removeGames uninstalls WrongGPU from every game that has it
// (managed, or with its own entry; refused at once while one runs); else each game keeps WrongGPU and gets its own uninstall entry back (Op::Detach). A
// game that fails stops it with *why, the launcher still installed with its settings. Then the entry, the shortcut, the
// launcher's own files (launcher.ini, launcher.log, logs, jobs), the shared %LOCALAPPDATA%\WrongGPU once no game uses it
// (the model data, the games' list, the logs), and the installed exe: moved out of its folder at once (it may be this
// running program; this process leaves the folder if it is its current directory), deleted by a hidden cmd.exe once it no
// longer runs. Never throws.
// A managed game whose record cannot be read is stale (its folder deleted by hand) and leaves the list, unless its drive is
// not reachable (a USB drive unplugged, a network share offline): then it is left as it is, stays on the list, and the
// shared data it reads stays too; its folder is added to *unreachable.
bool SelfUninstall(Context& ctx, const LauncherPaths& p, bool removeGames, const wstring& startMenu, Ui& ui, wstring* why, const GameOps& gameOps = {},
                   std::vector<wstring>* unreachable = nullptr);
// A game on this PC has WrongGPU (the launcher's list or its own Apps & features entry, with its record): the launcher's
// uninstall asks whether the games keep it only then.
bool AnyGameHasWrongGPU();
// What the launcher's uninstall finds before it asks (its question says what goes): read off the window's thread, since a game
// on a network share that is offline holds each look for the share's timeout. anyGame: as AnyGameHasWrongGPU; offlineGame:
// one of them is on a drive that is not connected (the shared data stays for it); portableUses: a portable copy uses the
// shared data (PortableLauncherUsesSharedData: it stays for that copy); modelData / rrData: the converted data that is there.
struct LauncherUninstallLook { bool anyGame = false, offlineGame = false, portableUses = false, modelData = true, rrData = false; };
LauncherUninstallLook LookBeforeLauncherUninstall(Context& ctx);
// The shared data removed by hand: a portable copy has no uninstall of its own, so its Settings offers this before its folder
// is deleted ("Remove data...", `remove-data`). The whole shared folder (%LOCALAPPDATA%\WrongGPU: the converted Model M and
// RR data, the notes and lists the games' uninstallers read) goes. Refused while a game has WrongGPU (it reads the model data
// when it runs; its last uninstall removes the data once no launcher uses it), and while the launcher is installed for this
// user (its own uninstall removes it). All or nothing: the folder is moved aside first, and a file in use there stops it with
// nothing removed. Never throws. look: what it finds (there: the folder is there; anyGame: as LookBeforeLauncherUninstall;
// installed: the installed launcher, or its settings, use the folder).
struct SharedDataLook { bool there = false, anyGame = false, installed = false, modelData = false, rrData = false; };
SharedDataLook LookAtSharedData(Context& ctx);
// ignore-files.txt stays (convert_ignore.h): *kept gets "kept ignore-files.txt (...)" and its path when there was one.
bool RemoveSharedData(Context& ctx, wstring* why, wstring* kept = nullptr);
// The hidden cleanups this process started (SelfUninstall's moved exe and folders) handed again for what is still there,
// as the program ends: they try for an hour, and a window may stay open longer than that. The window calls it last.
void RetryCleanupsAtExit(Context& ctx);

// ---- Play ------------------------------------------------------------------------------------------------------------
// The environment a game starts with: this process's, without LLVM_PATH, ROCM_PATH, HIP_PATH and HIP_DEVICE_LIB_PATH
// (they make the HIP runtime load another compiler and break it), plus `extra` (NAME=value).
std::vector<wchar_t> GameEnvironment(const std::vector<wstring>& extra = {});
// How Play started a game. A game in a Steam library whose app id its appmanifest gives (SteamAppId) starts through Steam,
// "steam://rungameid/<appid>" (the address Steam's own desktop shortcuts use: the game's normal start with the player's
// launch options, their Steam account, so its saves and Steam Cloud), when Windows knows the steam: address. Every other
// game, and a Steam game when that start fails, starts directly (its exe, GameEnvironment's environment, Windows'
// administrator prompt when its manifest asks for it). note: why a Steam game started directly (empty otherwise).
// rc6 L7: a game an Epic manifest holds (EpicAppName) starts through Epic the same way,
// "com.epicgames.launcher://apps/<AppName>?action=launch&silent=true" (the player's Epic account, its cloud saves and
// overlay), when Windows knows that address; else, or when that start fails, directly. note: why an Epic game started directly.
struct PlayWay { bool steam = false, epic = false; wstring appid, epicApp, note; };
// What PlayGame asks Windows for (tests replace them; Steam and Epic are never started by a test). Empty members: the real ones.
struct PlayStarters {
    std::function<bool()> steamKnown;                                   // the steam: address has a handler
    std::function<bool(const wstring& url, wstring* why)> openUrl;      // ShellExecuteEx of the address (Steam's or Epic's)
    std::function<bool(const wstring& exe, wstring* why)> direct;       // the direct start (PlayDirect)
    std::function<bool()> epicKnown;                                    // the com.epicgames.launcher: address has a handler
    std::function<wstring(const wstring& exe)> epicApp;                 // the game's Epic AppName (EpicAppName)
};
// The way Play starts exe, without starting it: through Steam when it has an app id and steamKnown says yes; else through
// Epic when epicApp (its Epic AppName) is given and epicKnown says yes.
PlayWay ChoosePlayWay(const wstring& exe, bool steamKnown, bool epicKnown = false, const wstring& epicApp = L"");
bool SteamAddressKnown();
bool EpicAddressKnown();
bool PlayDirect(const wstring& exe, wstring* why);
// Play: refused when the exe is missing or the game runs already (why says so); else started as ChoosePlayWay says, logged
// to log when given ("play: ... through Steam (app id N)" / "... through Epic (app X)" / "... directly"). how: the way it took
// (also when it failed).
bool PlayGame(const wstring& exe, wstring* why, PlayWay* how = nullptr, Context* log = nullptr, const PlayStarters& with = {});

// ---- opt-in update check --------------------------------------------------------------------------------------------
struct UpdateInfo { bool checked = false, newer = false; wstring version, page, message; };
// Off unless enabled and a URL is set; https only; one GET, at most 64 KB. The reply: a JSON release object
// ("tag_name", "html_url") or text lines "version=x.y.z" and "page=<url>".
UpdateInfo CheckForUpdate(const GlobalSettings& gs, const wstring& currentVersion);
bool ParseUpdateReply(const string& body, wstring& version, wstring& page);

}  // namespace wg
