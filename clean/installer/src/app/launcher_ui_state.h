// SPDX-License-Identifier: MIT
// WrongGPU launcher window: what launcher_ui.cpp (the window, its input and operations) shares with ui_activate.cpp (what a
// click, Enter or Space on a widget does): the window's messages, the messages its workers post, and the window's state that
// both use. Split from launcher_ui.cpp (the spec side's 256 KB scan limit) with no change in behaviour. Written for this project.
#pragma once
#include "app/launcher_window.h"

#include <atomic>
#include <memory>
#include <thread>

#include "wg_core/logs.h"
#include "wg_core/model.h"
#include "wg_core/optiedition.h"
#include "wg_core/optiscaler.h"
#include "wg_core/payload.h"
#include "wg_core/pe.h"
#include "wg_core/rr.h"

namespace wg::app {
using namespace wg::app::lui;
namespace win {

const UINT WM_APP_SCAN = WM_APP + 20, WM_APP_STEP = WM_APP + 22, WM_APP_OPDONE = WM_APP + 23, WM_APP_ART = WM_APP + 24,
           WM_APP_HEALTH = WM_APP + 25, WM_APP_UPDATE = WM_APP + 26, WM_APP_NOTE = WM_APP + 28,  // WM_APP + 21: launcher_window.h; 27, 29, 31, 35, 37: UI Automation (ui_uia.cpp)
           WM_APP_TASK = WM_APP + 30, WM_APP_ENTRIES = WM_APP + 32,
           WM_APP_UNINSTALL = WM_APP + 33,  // WM_APP_UNINSTALL: a second start from Apps & features
           WM_APP_RRPREP = WM_APP + 48,     // release 0.6.0: prepare Ray Reconstruction for the games waiting (ui_rrprep.cpp)
           WM_APP_RRBG = WM_APP + 49;       // release 0.6.0 polish: the background re-prepare's progress (ui_rrprep.cpp)
const int kEditSearch = 901, kEditUrl = 902, kEditOpti0 = 910;

struct ScanMsg { std::vector<GameEntry> games; };
struct StepMsg { int done, total; wstring text; };
// the game page's data read from the game's folder (ReadPageData; have: read)
// rrcfg: Borderlands 4 / Hogwarts Legacy's Engine.ini line (RrConfigLineState: -1 its folder unknown, 0 no line, 1 there, 2 there
// but not read-only; -2 another game), rrcfgFile: that file (UI review #2: the game page's file check), rrcfgFixed: the install
// record has WrongGPU's line (r2b #1: rr_config_fix=on without it waits for the game's settings folder)
// rrcfgWhy: why the file couldn't be read (rrcfg 3, r3 #8)
struct PageData { bool have = false; Profile saved; OptiView opti; std::vector<OutsideEdit> outside; int rrcfg = -2; wstring rrcfgFile; bool rrcfgFixed = false; wstring rrcfgWhy; };
// an operation's end: its result, the request, the game's name as the Library had it, and the game's entry (and its page's
// data) read again on the worker (the window's thread never hashes a game's files, nor waits for its disk)
struct OpDoneMsg { OpResult res; wstring dir, exe; Op op = Op::Install; OpRequest req; wstring name; GameEntry fresh; bool haveEntry = false; PageData page;
                  std::vector<std::pair<wstring, OpResult>> batched;  // Update all: the other queued games its elevated worker did (StartOp)
                  bool needsPermission = false; };  // r2c: the window's own apply (OpRequest::noPrompt) found the folder protected: nothing run
// games read again off the window's thread (a game that closed): their entries and their pages' data
// poll: timer 4's check; again: a second read of what a worker could not read (ReadAgainLater); recheck: Check again
// name: the game Check again read (its toast names it: it shows wherever the player is)
struct EntriesMsg { struct Item { wstring dir, exe; GameEntry fresh; PageData page; }; std::vector<Item> items; bool poll = false, again = false, recheck = false; wstring name;
                   std::vector<wstring> seen, closed; };  // rc6: games Play started seen running / closed by the watch (WatchPlayed)
// a game page opening (OpenGame): its data read on a worker; seq: which ask it answers (an older one landing late is dropped)
const UINT WM_APP_PAGE = WM_APP + 34, WM_APP_PLAY = WM_APP + 36;  // WM_APP_PLAY: "Apply and play" parked under a modal loop
const UINT WM_APP_PLAYED = WM_APP + 38;  // Play's worker: Windows started the game, or why not (PlayMsg)
// changes: the game folder's change count (DirChanges) when the page was asked for
struct PageMsg { unsigned seq = 0; GameEntry e; PageData page; unsigned changes = 0; };
// HealthMsg, FirstRunMsg: launcher_window.h
// a background task's end (RunTask): what the window does with its result, on its own thread
struct TaskMsg { std::function<void()> done; };

// ---- launcher_ui.cpp's state and helpers that ui_activate.cpp uses ------------------------------------------------------
extern bool firstRunBusy;
extern std::shared_ptr<ModelState> firstRunModel;
extern std::vector<wstring> firstRunAdopt;
extern bool rechecking, rrSetupNext, openedToUninstall, rescanAsked, attnReveal;
extern std::vector<int> attnShownBefore;
extern wstring playAfterApply, addedExe, addedFolder;
extern ULONGLONG clickGuardAt;
extern HANDLE* windowLock;
// Worker threads (a scan, the health read, a release check, an operation, a task): detached, counted while they run. The
// window's thread waits for them before the process ends (they use g.ctx and the log: RunLauncherWindow). A throw a worker
// did not catch itself never leaves its thread (that is std::terminate: the window gone without a word): it is logged here.
// The workers whose result the window waits for (a scan, the first run's checks, the health read, a release check) post one
// in words themselves first.
extern std::atomic<int> workers;
template <class F> void Worker(F f) {
    ++workers;
    std::thread([f]() mutable {
        try { f(); }
        catch (...) { try { LogLine(g.ctx, L"worker ERROR: unexpected failure"); } catch (...) {} }
        --workers;
    }).detach();
}
std::vector<wstring> AdoptableDirs();
void ApplyTheme();
bool Browse(int mode, wstring& out);  // ui_frame.cpp
void ConvertModelAgain(const wstring& dll, bool noScan, wstring& msg, bool& ok, bool& again);
void ConvertRr(const wstring& dll);
Ui& NoUi();
Ui QuietUi();
void OnAdopt(const GameEntry& e);
void OnApply();
void OnInstall(const GameEntry& e, Op op);
void OnLeftovers(const GameEntry& e, bool clean);
void OnOptiFix(bool keep);
void Play(const GameEntry& e);
bool ReadPageDataSafe(const GameEntry& e, const Profile& defaults, PageData& d);
int RunDialog(int icon, const wstring& body, std::vector<DialogButton> buttons, int cancel);
void RunTask(int kind, std::function<std::function<void()>()> work);
bool SaveSettings();
void SaveSettingsForInstalledCopy();
void SaveUrl(bool check);
void SayResultAs(int kind, const wstring& dir);
void ScrollBy(float dy);
void StartScan();
void TakeModelState(const HealthMsg& m);
void UpdateAll();
void WriteProfileOnly(const Profile& p);
void WriteProfileOnly(const Profile& p, bool noPrompt);  // noPrompt: OpRequest::noPrompt (the window's own apply, r2c)

}  // namespace win
namespace lui {
void FieldNow();        // launcher_ui.cpp
void FinishFirstRun();
void ActivateWidget(int id);  // ui_activate.cpp
}  // namespace lui
}  // namespace wg::app
