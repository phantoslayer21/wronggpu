// SPDX-License-Identifier: MIT
// WrongGPU launcher: what the launcher's command line (app/cli.cpp) and its window (app/launcher_ui.cpp) share: the
// launcher's paths, global settings and the engine set up for the launcher. Written for this project.
#pragma once
#include "app/app.h"
#include "wg_core/launcher.h"
#include "wg_core/rr.h"
#include "wg_core/rrprep.h"

namespace wg::app {

struct LauncherState {
    LauncherPaths paths;
    GlobalSettings global;
    wstring dataOverride;   // --data=
    wstring opLog;          // the log of the current operation (copied into the game's wronggpu-setup\setup.log)
    wstring firstRunLog;    // a downloaded copy's first start: its log in the temp folder until the first run's choice
};
extern LauncherState gl;  // app/cli.cpp

// Sets up g.ctx for the launcher (one Apps & features entry when the launcher is installed, HIP optional, the
// OptiScaler layout, the embedded restorer) and reads the paths and launcher.ini. worker: the elevated worker (no log in
// this account's folders; the job names the player's). window: the window's start (a downloaded copy's first run keeps
// its log in the temp folder until the first run's choice: firstRunLog).
void LauncherInit(const wstring& logOverride, bool worker = false, bool window = false);
// The first run's choice made (FinishFirstRun, Run portable): its log lines so far join launcher.log in the data folder the
// choice gave (gl.paths), and the log goes on there. Nothing to do when the log was there from the start.
void SettleFirstRunLog();
// The window closes before the first run's choice: its log in the temp folder goes (nothing is left of a start that
// changed nothing).
void DropFirstRunLog();
// The current operation's log: <data>\logs\last-operation.log, started empty; its lines are added to launcher.log
// when the operation ends (OpLogEnd).
void OpLogBegin();
void OpLogEnd();
// The elevated worker: writes the job, starts "WrongGPU.exe apply --job ... --result ..." with runas, waits, reads the
// result. Called on a worker thread by the window.
OpResult RunElevated(const OpRequest& r, HWND owner);
// Several operations in one elevated worker (one permission prompt: Update all's games in protected folders), in order;
// stop: none after the first failure. A result per request (kOpNotRun: not run).
std::vector<OpResult> RunElevatedOps(const std::vector<OpRequest>& rs, HWND owner, bool stop);
// 0.5.x installs (and managed games) adopted: in this process where the folder allows it, the ones in protected folders
// together in one elevated worker (one permission prompt) when a window owns the prompt. kept: the names of the games that
// were not adopted; results: one per folder.
void AdoptGames(const std::vector<wstring>& dirs, HWND elevateOwner, std::vector<wstring>* kept, std::vector<OpResult>* results = nullptr);
// After the Ray Reconstruction data was converted (an install, update or repair, Convert): the settings files of the RR
// games installed before it (but `except`'s) brought to the data's state; the ones this process may not write (protected
// folders; `known`: already found by the caller) get an Apply in one elevated worker when a window owns the prompt
// (elevateOwner). Each one still not written: a warning (kRrNotUpdated + the game's name ...), RR stays off there.
const wchar_t* const kRrNotUpdated = L"Ray Reconstruction is ready, but the settings of ";
void SyncRrSettings(HWND elevateOwner, const wstring& except, std::vector<wstring>& warnings, const std::vector<RrSettingsFailure>* known = nullptr);
// The launcher installed for this user (Settings, the first run and `self-install`): SelfInstall, then every managed game
// and every folder of alsoAdopt (the 0.5 installs the window found) adopted (its own Apps & features entry folded into the
// launcher's: AdoptGames), and `settings` (the caller's copy, taken when it started) saved for the installed copy. kept:
// the games that keep their own entry (an adoption failed). False with `why` when the launcher could not be installed.
bool InstallLauncherForUser(const wstring& dir, const wstring& startMenu, HWND elevateOwner, const GlobalSettings& settings, const std::vector<wstring>& alsoAdopt,
                            wstring& why, std::vector<wstring>* kept);
// The launcher uninstalled (Settings, Apps & features through the window, `self-uninstall`): SelfUninstall with each game's
// operation elevated where the folder needs it when a window owns the prompt (elevateOwner; nullptr: in this process).
// unreachable: the managed games left as they are because their drive is not connected (SelfUninstall).
bool UninstallLauncher(bool removeGames, const wstring& startMenu, HWND elevateOwner, Ui& ui, wstring& why, std::vector<wstring>* unreachable = nullptr);
// Whether an operation gives the game its own Apps & features entry: only while no launcher is installed for this user. Read
// again before each of the window's operations and after the launcher's uninstall (a portable window stays open meanwhile,
// and the launcher may be installed or uninstalled from elsewhere), never kept from the process's start.
void RefreshPerGameEntry();

int RunCli(int argc, wchar_t** argv);  // a verb (app/cli.cpp)
// The launcher window (app/launcher_ui.cpp). uninstall: opened by Apps & features ("WrongGPU.exe uninstall-launcher"):
// Settings with the uninstall question (an open window is brought forward and asks it instead).
int RunLauncherWindow(bool uninstall = false);
// The window built hidden (never shown): every page's controls as JSON into dumpFile, optional BMPs (app/launcher_ui.cpp).
// apply (tests; the game's files are written): "fps" the checklist's FPS overlay button and Apply on the first game with
// WrongGPU; "rrconfig" the Engine.ini setting's WrongGPU, then Myself, each with Apply, on Borderlands 4 / Hogwarts Legacy
int RunLauncherUiDump(const wstring& dumpFile, const wstring& shotDir, bool noScan, const wstring& apply = L"");
// Development screenshots of the window from fixture game records (tests/ui_fixture.ini): PNGs of every page at w x h
// DIPs for each scale (1 = 96 DPI) into outDir. Nothing is shown; no game folder is read (Steam's art cache is).
int RunLauncherUiRender(const wstring& outDir, const wstring& fixtureFile, int w, int h, const std::vector<float>& scales, const wstring& frames,
                        const wstring& theme, wstring* why);  // theme: light (default) or dark

}  // namespace wg::app
