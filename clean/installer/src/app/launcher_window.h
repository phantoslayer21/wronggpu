// SPDX-License-Identifier: MIT
// WrongGPU launcher window: the window's own helpers (launcher_ui.cpp, namespace wg::app::win) that its offscreen frames
// (ui_render.cpp, "WrongGPU.exe ui-render") use. Written for this project.
#pragma once
#include "app/ui_shell.h"

#include <shared_mutex>

#include "wg_core/model.h"

namespace wg::app::win {
// rc6: a game Play started (ui_play.cpp `played`, by exe in lower case, until the running-game watch saw it close): when, and
// whether the watch has seen it running yet (WatchPlayed, wg_core/checks.h)
struct PlayedGame { ULONGLONG at = 0; bool seen = false; };
extern std::map<wstring, PlayedGame> played;
// ui_rrprep.cpp (release 0.6.0): Ray Reconstruction's kernels prepared by the launcher (wg_core/rrprep.h)
bool RrGameInstalled(const wstring& dir);
bool RrReadyForGame(const wstring& dir, const wstring& exe, wstring* why = nullptr);
bool PrepareRrNow(const wstring& dir, const wstring& exe);  // false: the window is closing
bool RrReadyBeforePlay(const wstring& dir, const wstring& exe);
void QueueRrPrep(const wstring& dir, const wstring& exe);
void RunRrPrepQueue();
bool RrPrepRunningNow();
void CancelRrPrep();
// release 0.6.0 polish: RR's kernels prepared again in the background after a driver / WrongGPU update (ui_rrprep.cpp)
void StartRrRecheck();
void StopRrBackground();
void RrBackgroundMessage(WPARAM wp);
wstring RrBackgroundStatus();                   // the toolbar status line's part (empty: not running)
wstring RrBackgroundWords(const wstring& exe);  // the game page's RR row while the background run has the game (empty: not)
bool RrPrepCancelled();
HWND CreateLauncherWindow(bool live);                  // the window (live) or a message-only one (ui-dump, ui-render)
bool RenderFrame(float w, float h, float scale, const wstring& png);  // one offscreen frame (measured, then drawn; png: saved there)
void ShowEntry(const GameEntry& e, bool keepView);     // a game's page with this entry
void SetHealth(bool model, const GpuInfo& gpu);        // the model data's state and the graphics card (the toolbar's line)
void OpenMoreMenu();                                   // the game page's More menu
void SplitDialogText(const wstring& text, wstring& title, wstring& body);
wstring PlayUnsavedText(const wstring& name, bool busy = false);
std::vector<lui::DialogButton> PlayUnsavedButtons(bool busy = false);
// The operation running now was started by a control whose result has words of its own (the RR row's Repair, the
// outside-change note's two buttons): which (launcher_ui.cpp SayResultAs); the success in words (ui_words.cpp; fresh: the
// game's entry read again after it, null: not read), and a core step in the page's words ("" : the verb alone). RES_GONE: Remove
// files on a game that is gone (Orphaned): its question, verb, result and failure say that, not "uninstalled"
enum { RES_PLAIN, RES_RRREPAIR, RES_KEEP, RES_REVERT, RES_GONE };
wstring OpSuccess(Op op, const OpResult& res, const OpRequest& req, const GameEntry* fresh, const wstring& name, int as = RES_PLAIN);
wstring StepWords(const wstring& s, int done, int total);
wstring StartSaid(const OpRequest& r, const wstring& step, bool rrSetup, const wstring& who);  // an operation's start, for Narrator
wstring ApplyWaitsWords(const GameEntry& e);  // Ctrl+S while the capsule's Apply waits: why, in words (a toast)
wstring OpFailLead(Op op, int as = RES_PLAIN);
wstring OpFailTitle(Op op, const wstring& name, int as = RES_PLAIN);
wstring PlainCause(const wstring& raw, int code = 0);
// The cause of a failed launcher task in plain words (ui_words.cpp): never a game's folder or the game's advice. kCauseLauncher:
// Install for this user (the launcher's own folder, Start, Installed apps); kCauseLauncherUninstall: its uninstall (a game
// that stopped it is said as that game's); kCauseLogs: Collect logs.
const int kCauseLauncher = 0, kCauseLauncherUninstall = 1, kCauseLogs = 2;
wstring PlainTaskCause(const wstring& raw, int what);
// ui_words.cpp: the result words' helpers (a capital and a full stop; a case-blind "contains"; a warning in plain words,
// empty when it is not for the player)
wstring Sentence(wstring s);
bool Has(const wstring& text, const wchar_t* part);
wstring PlainWarning(const wstring& w);
// the games the launcher's uninstall left as they are (their drive not connected; names as said), and what to do: "" for none
wstring OfflineGamesWords(const std::vector<wstring>& names, bool removeGames);
// ui_words.cpp, the install's own words: the launcher's uninstall done (as asked; the games left as they are; other games
// done besides them), Repair's question on an older WrongGPU (rrSetup: the RR row's Set up), Update all's games left out
// (name, gone since the scan), "installed" with the games that kept their own entry, the 0.5 installs not adopted
wstring LauncherUninstalledText(bool anyGame, bool games, bool here, const std::vector<wstring>& offline, bool others);
wstring RepairUpdateText(const wstring& name, const wstring& from, const wstring& to, bool rrSetup);
wstring PresetKText(const wstring& name, const wstring& to);
wstring SkippedWords(const std::vector<std::pair<wstring, bool>>& skipped);
wstring InstalledWords(const std::vector<wstring>& kept, const wstring& lead);
wstring NotAdoptedWords(const std::vector<wstring>& kept);
wstring LogsSavedText(const wstring& zip, bool desktop);
wstring LogsSavedText(const std::vector<wstring>& zips, const std::vector<wstring>& names, bool desktop);  // one file per game
// the launcher's uninstall question (what it found; its own window; logs: Collect logs offered, answer 3) and its buttons
wstring UninstallLauncherText(const LauncherUninstallLook& look, bool here, bool logs);
std::vector<lui::DialogButton> UninstallLauncherButtons(bool anyGame, bool logs);
wstring UninstallGameText(const wstring& name, bool gone, int data);  // a game's Uninstall question (gone: Remove files)
wstring RemoveDataText(bool model, bool rr);  // a portable copy's Remove data... question (the converted data that is there)
// launcher_ui.cpp: a read on a worker while the window answers (false: it is closing), a note with Show details;
// ui_words.cpp: Collect logs from the launcher's uninstall question (one file per game with WrongGPU, then where they are)
bool WaitForRead(const wstring& dir, const wstring& verb, int task, std::function<void()> work);
void InformDetails(const wstring& text, int icon, const wstring& details);
void CollectLogsBeforeUninstall();
bool FirstRunPortable();  // ui_words.cpp: the first run's "portable" choice (false: refused, a note said why)
void HandOverToInstalled(HANDLE& one, const wstring& installedExe, const wchar_t* why);  // ui_words.cpp: a newer download, its question answered
bool AnswerReplaceOffer(HANDLE& one, const wstring& installedExe, const wstring& have, const LauncherStartDecision& start);  // ui_words.cpp: "Update WrongGPU to X?"; true: Not now, this copy runs (rc5)
void RemoveDataAsked();  // ui_words.cpp: a portable copy's Settings > Remove data... (RemoveSharedData)
void StartHealth(bool gpu);  // the model data, the RR data and the graphics card read again (the status line, Settings)
// Clean up / Adopt's question; otherSetup (0.6.1): the files are used by another mod's setup (said, Cancel the default)
wstring LeftoversText(const wstring& name, const std::vector<Leftover>& files, bool clean, bool otherSetup = false);
std::vector<lui::DialogButton> LeftoversButtons(bool clean, bool otherSetup = false);
// ui_dump.cpp ("WrongGPU.exe ui-dump") drives the window offscreen with these
const float kDefaultW = 1180, kDefaultH = 724;           // the window's first size (DIPs)
extern bool firstRunBusy, focusFirstStop;              // the first run's checks still running; the Library's first stop takes the focus next
extern bool revealFocus;                               // the focus is brought into view after the frame (the window's paint)
void StartFirstRun();
// ui_checks.cpp: the first run's checks (StartFirstRun runs them on a worker; their lines and findings come back as
// WM_APP_FIRSTRUN). WM_APP + 21 is in launcher_ui.cpp's list of the window's messages.
const UINT WM_APP_FIRSTRUN = WM_APP + 21;
// the model data's state (off the UI thread: it reads and hashes the model file), and the graphics card when asked
// seq: when its read began (ReadModelState; 0: not a read, always taken): an older read that ends later is not taken
struct HealthMsg { bool model = false; wstring why, source, hip; bool gpuDone = false; GpuInfo gpu; bool rr = false; wstring rrWhy, rrSource; unsigned seq = 0; };
// the first run's checks (only read: nothing on this PC changes before the player chooses and presses Continue; state: 0 ok,
// 1 note, 2 problem): posted as each part finishes (WPARAM 1, lines only) and once at the end with the games, the model
// data's state read on the worker (the window's thread never hashes the model file), the DLSS file found and converted in
// memory (written on Continue) and the 0.5 installs Continue adopts
struct FirstRunMsg {
    std::vector<std::pair<int, wstring>> lines;
    std::vector<GameEntry> games;
    HealthMsg health;
    std::shared_ptr<ModelState> model;
    std::vector<wstring> adopt;
};
extern std::shared_mutex optiLock;                     // an install preparing the OptiScaler package holds it alone (launcher_ui.cpp)
void ReadModelState(HealthMsg& m);
void FirstRunChecks(HWND hwnd, const GlobalSettings& gs, const wstring& dll, bool dump);
// ui_checks.cpp: a game that uses the Model M / Ray Reconstruction data is running (one look at the processes, no game
// folder read): said in a note, true: nothing starts
bool ModelGameRunningSaid();
bool RrGameRunningSaid();
void RefreshModelNow();                                // the model data's state read here (no worker)
void SetGpu(const GpuInfo& gpu);
void OnKey(WPARAM k, LPARAM lp);
bool Navigate(lui::Page p);
std::vector<const lui::Hit*> Focusables();
void FocusHit(const lui::Hit* h);
wstring GameInfo(const GameEntry& e);
wstring OutsideText();
wstring OptiInfoText(const lui::OptiView& o);
wstring OptiOutsideText();
// ui_uia.cpp: the keyboard focus on the Library's places (a lockup's ids are its game's place in the list; the banner's name
// whichever game it shows), kept with the game the player chose when the list or the banner changes under it
extern std::vector<int*> focusUnderDialogs;            // the focus kept under each open dialog (RunDialog), innermost last
wstring PlaceGame(int id);                             // the game a place names as last drawn (folder|exe, lower case), else empty
void HoldPlaceFocus();                                 // before the game list changes: the places' games (now and under dialogs)
void RestorePlaceFocus();                              // after it: each by its game; a game that left: -1 (the nearest stop)
void FollowPlaceGame(int lastId, const wstring& lastGame);  // in a Library frame: the focus that names another game goes with its game
void RevealIfMoved(const lui::Hit* at, bool samePage);  // the focused control moved on the page, not by a scroll: into view
// ui_uia.cpp: the keyboard model's own window messages (WndProc asks first; true: handled, r its result): a lone Alt opens
// no invisible system-menu mode, Browser Back goes back, Space acts as it comes up on the control it went down on
extern int spacePress;                                 // the control Space went down on (OnKey), 0 none
void SpaceUp();
bool AccessMessage(UINT msg, WPARAM wp, LPARAM lp, LRESULT& r);
void FocusOptiField(const string& name, const wstring& why);  // Apply refused a value: the focus on its row's field, its text selected, marked invalid
// ui_frame.cpp: a text field's EDIT messages after the window's own keys (EditProc): an empty field paints its placeholder
// in the theme's text2, as FieldBox draws it (not the EDIT's cue banner in a fixed system grey), then DefSubclassProc
LRESULT EditCueProc(HWND h, UINT m, WPARAM wp, LPARAM lp);
void KeyApply();                                       // Ctrl+S, Return in an OptiScaler number field: the capsule's Apply (the keys stay in the field)
bool OnTabPage(int tab);                               // the toolbar tab's page is the open one (not a game's page kept)
// ui_rrconfig.cpp: after an install, repair or update of Borderlands 4 or Hogwarts Legacy whose rr_config_fix is still ask,
// the question about the game's Engine.ini once the window is free (wg_core/rrconfig.h)
void AskRrConfigLater(const wstring& dir, const wstring& exe, const Profile& saved);
bool RrConfigAskable(const wstring& dir, const wstring& exe);  // r7 #5: only with WrongGPU's RR files in the game
// "Fix it for me" chosen before the game made its settings folder (wg_core/rrconfig.h: nothing written then): the choice applied
// once the folder is there, when the window is free, once per game a session (r2b #1). Now: started at once if it can be (the
// ui-dump; true: the Apply started).
void ApplyRrConfigWhenDue(const wstring& dir, const wstring& exe);
bool ApplyRrConfigDueNow(const wstring& dir, const wstring& exe);
// r2c: the game's folder needs administrator permission, so the window's own apply wasn't run (WM_APP_OPDONE): its card says
// "Ready to add" (Fix it now), and the window doesn't try again this session. Forget: the ui-dump's second try (the worker's check)
void RrConfigNeedsClick(const wstring& dir);
void ForgetRrConfigDue(const wstring& dir);
void RrConfigAction(int id);                  // the Engine.ini steps' Open folder, Copy, Check again
void ShowRrConfigSteps(const wstring& dir);  // "Show me how": the game's page at its Engine.ini steps, the keyboard on Open folder
// r3 #5: the toast the launcher shows once when a protected game's Engine.ini line is ready to add (RrConfigReadyToAdd);
// empty without one
wstring ReadyToAddToast();
// The Engine.ini question answered (1 Fix it for me, 2 Show me how), as its dialog does (the ui-dump; r3 #6)
void AnswerRrConfigQuestion(const wstring& dir, const wstring& exe, const Profile& saved, int choice);
// the question's text for that game (RrConfigNotice; in a protected folder it says Windows asks for permission first: r4 #9)
wstring RrConfigQuestionText(const wstring& dir, const wstring& exe);
}  // namespace wg::app::win
