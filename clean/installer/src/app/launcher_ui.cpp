// SPDX-License-Identifier: MIT
// WrongGPU launcher: the window (build/launcher-design.md section 3; the "Bench" look of build/launcher/ui-mockup2,
// light and dark, following Windows' app mode unless Settings picks one): the Library of games as a rack of bays, a page per game (Install / Update / Repair / Uninstall / Roll back / Play, Clean up /
// Adopt for leftover files, the Basic settings, the last run with its warnings, WrongGPU's OptiScaler setup and what
// else is in the game folder), Settings, the first-run summary and the dialogs. Painted with Direct2D / DirectWrite
// (ui_*.cpp); native EDIT children for typed text; UI Automation from ui_uia.cpp. Long operations run on a worker
// thread; a game folder that needs administrator rights goes to the elevated worker (a job file and "WrongGPU.exe
// apply" started with runas). ui-dump (the launcher tests) and ui-render (fixture screenshots) paint the same frames
// offscreen, without a visible window. Written for this project.
#include "app/launcher_ui_state.h"

#include <dwmapi.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>
#include <windowsx.h>
#include <tlhelp32.h>

#include <commctrl.h>

#include <atomic>
#include <shared_mutex>
#include <thread>

#include "wg_core/json_writer.h"
#include "wg_core/logs.h"
#include "wg_core/model.h"
#include "wg_core/optiedition.h"
#include "wg_core/optiscaler.h"
#include "wg_core/payload.h"
#include "wg_core/pe.h"
#include "wg_core/rr.h"
#include "wg_core/rrconfig.h"

namespace wg::app {
using namespace wg::app::lui;

namespace win {


Gfx gfx;
bool firstRunBusy = false;
// What the first run's checks found and its Continue carries out with the player's choice: the model to write, the 0.5
// installs to adopt
std::shared_ptr<ModelState> firstRunModel;
std::vector<wstring> firstRunAdopt;
// Apps & features' Uninstall arrived while the window was busy or asking something: it starts once that ends
bool uninstallWaiting = false;
// The one-window lock RunLauncherWindow holds (its handle there), for a hand-over to the installed copy from inside the
// window (the first run's Install and continue)
HANDLE* windowLock = nullptr;
// Modal loops (a dialog, a file dialog) open now, and the notes that arrived on their own meanwhile (InformLater): their
// text, icon and the technical text behind their "Show details" (empty: none)
int modalDepth = 0;
struct Note { wstring text; int icon = 2; wstring details; };
std::vector<Note> pendingNotes;
// Close pressed during an operation that then ended well with notes to read (its warnings, the games Update all left out):
// the window closes once the last of them is closed (WM_APP_NOTE), not under them
bool closeAfterNotes = false;
// Background tasks that ended while a modal loop was open (RunTask's WM_APP_TASK): carried out once it closes, so their
// notes never open inside a file dialog's loop (on its disabled owner) or under a question
std::vector<std::unique_ptr<TaskMsg>> parkedTasks;
// A game's page being opened (OpenGame): the latest ask (an answer to an older one is dropped), the one still waiting (0:
// none; the pointer shows it works), the page it was asked on (it opens only if the player is still there), and an answer
// that arrived while a dialog or a file picker was open (it opens once that closes)
unsigned pageSeq = 0, pageWaiting = 0;
int pageAskedOn = 0;
// the game asked: shown before its files are read once they take over kPageReadMs (WM_TIMER 10, ShowReadingPage)
GameEntry pageAsked;
int pageAskedIndex = -1;
constexpr UINT kPageReadMs = 120;
// How often each game folder (lower case) has been changed under the window since it started: an operation that ended
// (WM_APP_OPDONE), a game read again after it closed or by Check again (WM_APP_ENTRIES). A game's page read on a worker
// (OpenGame) that one of these overtook may hold the files as they were before it, or half way through it: it is read
// again (WM_APP_PAGE), so the page never shows, and a later Apply never writes back, settings the operation replaced.
std::map<wstring, unsigned> dirChanges;
unsigned DirChanges(const wstring& dir) { const auto it = dirChanges.find(Lower(dir)); return it == dirChanges.end() ? 0u : it->second; }
void DirChanged(const wstring& dir) { ++dirChanges[Lower(dir)]; }
std::unique_ptr<PageMsg> parkedPage;
// "Apply and play": a game whose Apply ended while a dialog or a file picker was open starts once that closes (its note, if
// it can't start, never opens inside that loop): its folder and exe
wstring parkedPlayDir, parkedPlayExe;
// The modal loop just closed: the parked tasks' ends first, then the notes that waited, a page that was read, a game to start
void ModalClosed() {
    if (modalDepth > 0 || !v.hwnd) return;
    if (!parkedTasks.empty()) PostMessageW(v.hwnd, WM_APP_TASK, 0, 0);
    if (!pendingNotes.empty()) PostMessageW(v.hwnd, WM_APP_NOTE, 0, 0);
    if (parkedPage) PostMessageW(v.hwnd, WM_APP_PAGE, 0, 0);
    if (!parkedPlayExe.empty()) PostMessageW(v.hwnd, WM_APP_PLAY, 0, 0);
}
// Apps & features' Uninstall that waited: now, once nothing is busy and no question is open. It stays waiting until its
// message arrives (wParam 1), so a Close that comes first asks it instead of dropping it (WM_CLOSE)
void UninstallIfWaiting() {
    if (!uninstallWaiting || !v.live || !v.hwnd || v.busy || firstRunBusy || modalDepth > 0) return;
    PostMessageW(v.hwnd, WM_APP_UNINSTALL, 1, 0);
}
// a newer release's note among them (its icon kNoteUpdate): the release page its button opens
const int kNoteUpdate = 100;
wstring pendingUpdatePage;
// The games an operation finished on while a scan ran, with their entries as the operation left them: put over the scan's
// (older) result when it arrives
struct DoneEntry { wstring dir, exe; GameEntry fresh; bool have = false; };
std::vector<DoneEntry> doneSinceScan;
// Timer 4's check of the running games (are they still running; those that closed read again) is on a worker now: one at
// a time
bool pollingRunning = false;
bool rechecking = false;  // Check again (a game page's callout) is reading that game's folder on a worker
bool rrSetupNext = false;  // the next Repair is the RR row's Set up: its verb is "Setting up..." (StartOp)
bool openedToUninstall = false;  // the window opened by Installed apps' Uninstall, its question not answered yet (Cancel closes it)
bool replaceAsking = false;  // a newer download's window, only for "Update WrongGPU to X?": it saves no list or place (WM_DESTROY)
// Update all: the games it left out (name; true: gone since the scan, false: its files couldn't be read, its note said so),
// said at its end; and the results of the games an earlier game's elevated worker did with it (by folder, lower case)
std::vector<std::pair<wstring, bool>> queueSkipped;
std::map<wstring, OpResult> batchedResults;
// Update all: the games whose folder needs administrator rights (by folder, lower case), found on the worker that read each
// game before its questions (PrepareOp), so the queue puts them last together without a write test on the window's thread
std::map<wstring, bool> queueProtected;
wstring QueueSkippedWords() { const wstring w = SkippedWords(queueSkipped); queueSkipped.clear(); return w; }  // said once
// An Apply that failed after the player had left its page (Back, a tab, Ctrl+F with Apply chosen): the values they set, by
// game folder (lower case), put back as unsaved changes when that page opens again (never lost)
std::map<wstring, View::PendingApply> keptEdits;
// Play with unsaved changes, "Apply and play": the game (its exe) started once its Apply has written them
wstring playAfterApply;
// Add game / Add a folder: what was added, said once the scan that lists it lands
wstring addedExe, addedFolder;
// Add game's file out of launcher.ini again (it is not a game, or the scan that would say so failed)
bool SaveSettings();
static void DropAddedExe() {
    auto& added = gl.global.addedGames;
    const size_t n = added.size();
    added.erase(std::remove_if(added.begin(), added.end(), [](const wstring& x) { return IEq(FullPath(x), FullPath(addedExe)); }), added.end());
    if (added.size() != n) SaveSettings();
}
bool revealFocus = false;  // the focus moved to a control with the page (Back, a lockup): into view after the next frame
bool focusFirstStop = false;  // the first run ended: the Library's first stop gets the focus once it is drawn (PaintAll)
// "Show N more" pressed from the keyboard: the lockups shown before (game indexes' ids); the first one it revealed takes the
// focus once drawn (PaintAll), not the button, which moved below them all
bool attnReveal = false;
std::vector<int> attnShownBefore;
// the Library's lockups (game indexes) in reading order as last drawn: Remove from Library puts the focus on its neighbour
std::vector<int> libraryOrder;
void FocusHit(const Hit* h);
int lastSettingId = 0;  // the game page's control a setting was last changed with (Activate): the focus goes back to it when the capsule closes
bool rescanAsked = false;  // Rescan was pressed: the scan's outcome is said when it lands (WM_APP_SCAN)
// The native text fields' font (all in the UI font: a search, an address, a number are values, not code) and its line height in
// pixels (the fields' height), measured once per font
HFONT editFont{};
int editFontH = 0;
HBRUSH editBrush{};
HICON bigIcon{}, smallIcon{};
// The window's last normal place (screen pixels, kept as it moves and sizes while neither maximized nor minimized) and the
// way it starts (maximized as it closed, or as its shortcut's "Run" asks): written into launcher.ini as it closes
RECT normalRect{};
int startShow = SW_SHOWNORMAL;
void SavePlacement(HWND hwnd);
const float kWheelLine = 32;  // DIPs a wheel "line" scrolls (a page, a dialog's text, the first-run card): Windows' 3 lines a notch = 96
// What a wheel notch scrolls as the player set it (Settings > Mouse: lines at a time, or one screen at a time): `screen` DIPs
// for one screen; read on each turn (the setting can change while the launcher runs)
float WheelNotch(float screen) {
    UINT lines = 3;
    if (!SystemParametersInfoW(SPI_GETWHEELSCROLLLINES, 0, &lines, 0)) lines = 3;
    return lines == WHEEL_PAGESCROLL ? screen : std::min(lines, 100u) * kWheelLine;
}
UINT dpi = 96;
// A double click whose first click closed a dialog or a menu (or opened the next dialog in its place) is not a second
// press on whatever is under the pointer now: when the last one opened or closed
ULONGLONG clickGuardAt = 0;

// Worker threads (a scan, the health read, a release check, an operation, a task): detached, counted while they run. The
// window's thread waits for them before the process ends (they use g.ctx and the log: RunLauncherWindow).
std::atomic<int> workers{0};
// A throw a worker did not catch itself never leaves its thread (that is std::terminate: the window gone without a word): it
// is logged here. The workers whose result the window waits for (a scan, the first run's checks, the health read, a release
// check) post one in words themselves first.
// Worker (the template): launcher_ui_state.h
// g.ctx.opti (the player's OptiScaler package) is written only by an operation's worker (InstallOp prepares it, OptiTempDir
// names its folder once); the workers that inspect games read it. They hold this lock shared, an operation that may write it
// holds it alone (one runs at a time: it reads without the lock). Never taken on the window's thread.
std::shared_mutex optiLock;
bool OptiMayChange(const OpRequest& r) {
    if (r.op != Op::Install && r.op != Op::Update && r.op != Op::Repair) return false;
    const OptiPackage& o = g.ctx.opti;
    return (!r.optiPackage.empty() && (!o.ok() || !IEq(o.source, FullPath(r.optiPackage)))) || o.tempDir.empty();
}

// Windows shutting down, signing out or restarting while an operation writes a game's files: asked to wait (the reason shows
// on the shutdown screen) until it ends
void BlockShutdown(bool on) {
    if (!v.live || !v.hwnd) return;
    if (on) ShutdownBlockReasonCreate(v.hwnd, L"WrongGPU is changing a game's files. This takes a moment.");
    else ShutdownBlockReasonDestroy(v.hwnd);
}
// The work running now only reads (and so never holds up Windows' shutdown or sign-out with the reason above): collecting
// logs, the "Preparing..." look at a game's files before an install's questions (InspectAsync), the first run's checks
bool inspecting = false;  // InspectAsync's look at a game is running
bool WritesNow() { return v.busy && v.task != T_COLLECT && !inspecting; }

COLORREF Ref(Col c) { return RGB((int)std::lround(c.r * 255), (int)std::lround(c.g * 255), (int)std::lround(c.b * 255)); }
// ui_frame.cpp: the window's frame (the title bar's colours, its icon, its place between runs) and the offscreen frames
void ThemeTitleBar();
void SetWindowIcons();
bool SavedPlacement(RECT& r, HMONITOR& mon, UINT& mdpi, bool& maxed);
// The frame's thickness in pixels (the resize border Windows keeps at the top of the client area; maximized, the part of
// the window that hangs off the screen's edge)
int FrameY() { return GetSystemMetricsForDpi(SM_CYFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi); }
int FrameX() { return GetSystemMetricsForDpi(SM_CXFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi); }
// The theme of the launcher's "appearance" setting (system: Windows' app mode), applied to the whole window.
void ApplyTheme() {
    SetAccent(AccentFromKey(gl.global.accent), WindowsAccentRgb());
    SetTheme(ThemeIsDark(gl.global.appearance));
    if (editBrush) { DeleteObject(editBrush); editBrush = nullptr; }
    ThemeTitleBar();
    if (v.hwnd && v.live) {
        for (auto* e : {&v.edSearch, &v.edUrl}) if (e->h) InvalidateRect(e->h, nullptr, TRUE);
        for (auto& e : v.edOpti) if (e.h) InvalidateRect(e.h, nullptr, TRUE);
    }
    Invalidate();
}

Ui& NoUi() { static Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); }; return ui; }
// The same for a worker thread (its own, not the shared one above)
Ui QuietUi() { Ui ui; ui.step = [](int, int, const wstring&) {}; ui.log = [](const wstring& s) { LogLine(g.ctx, s); }; return ui; }
int Px(float dip) { return (int)std::lround(dip * dpi / 96.f); }

bool Browse(int mode, wstring& out);  // ui_frame.cpp: the file and folder pickers (0 exe, 1 folder, 2 model dll, 3/4 OptiScaler, 5 RR dll)

// ---- dialogs (custom-drawn, modal: a nested message loop as MessageBox has) ---------------------------------------
// A short first paragraph is the dialog's title, the rest its text ("Uninstall WrongGPU from X?\n\nThe game's ...").
void SplitDialogText(const wstring& text, wstring& title, wstring& body) {
    const size_t p = text.find(L"\n\n");
    if (p != wstring::npos && p <= 110) { title = text.substr(0, p); body = text.substr(p + 2); }
    else if (p == wstring::npos && text.size() <= 90 && text.find(L'\n') == wstring::npos) { title = text; body.clear(); }
    else { title.clear(); body = text; }
}
// Re-entrant: a dialog opened while another is open (a note that could not wait) is shown over it; when it closes the one
// under it comes back with its own answer still to give, and the keyboard focus goes back where it was.
int RunDialog(int icon, const wstring& body, std::vector<DialogButton> buttons, int cancel) {
    if (!v.live || !v.hwnd) return cancel;
    const Dialog outer = v.dialog;
    int outerFocus = v.focus;  // a scan that lands meanwhile moves it with its game (focusUnderDialogs)
    focusUnderDialogs.push_back(&outerFocus);
    const bool outerVisible = v.focusVisible;
    // a text field that had the keys (or was to get them once in view): they go back to it when the dialog closes
    const bool outerKeys = EditOf(outerFocus) && (GetFocus() == EditOf(outerFocus) || v.editFocusId == outerFocus);
    ++modalDepth;
    v.menu.open = false;
    v.dialog = Dialog();
    v.dialog.open = true; v.dialog.icon = icon; v.dialog.buttons = std::move(buttons); v.dialog.cancel = cancel;
    v.dialog.at = NowMs();
    clickGuardAt = GetTickCount64();  // the second click of a double click that opened it does not answer it
    SplitDialogText(body, v.dialog.title, v.dialog.body);
    v.dialog.result = cancel;
    // the default button (Apple HIG, Alerts): for a question about an action the player chose, the action, so Return
    // confirms it; a note's single button. The default shows by its fill. A warning (icon 1, 3) or a destructive action
    // (Uninstall, Remove from games too) has no filled default (its buttons are drawn as equals, the action never looks like
    // Return's answer): the keyboard focus is on Cancel, so Return dismisses it safely, as a Windows destructive
    // confirmation and a macOS alert do. Either way the ring shows on the focused button when the keyboard was in use (a
    // dialog opened from the keyboard), as Windows' dialogs show their focus cue. Escape cancels either way.
    // UI Automation hears the dialog open (UiaFrame: its sheet, the focus, its question announced) and the focus come back.
    v.focus = 0;
    if (const int def = DialogDefault(v.dialog); def >= 0) v.focus = ID_DIALOG + def;
    else for (size_t i = 0; i < v.dialog.buttons.size(); ++i) if (v.dialog.buttons[i].result == cancel) v.focus = ID_DIALOG + (int)i;
    v.focusVisible = outerVisible && v.focus != 0;
    Invalidate();
    UpdateWindow(v.hwnd);  // its buttons' hits now, not the last frame's (a dialog that replaces another at the same place)
    MSG m;
    while (v.dialog.open) {
        const BOOL r = GetMessageW(&m, nullptr, 0, 0);
        if (r == 0) { PostQuitMessage((int)m.wParam); break; }  // the outer loop ends too
        if (r < 0) break;
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    const int result = v.dialog.result;
    focusUnderDialogs.pop_back();
    v.dialog = outer;  // the dialog under this one, if any (else closed)
    v.focus = outerFocus;
    v.focusVisible = outerVisible;
    if (outerKeys) v.editFocusId = outerFocus;  // the field takes the keys again as soon as it shows (SyncEdits)
    clickGuardAt = GetTickCount64();  // the second click of the double click that closed it does not press what is under it
    --modalDepth;
    ModalClosed();
    UninstallIfWaiting();
    Invalidate();
    return result;
}

// ---- the game page's data --------------------------------------------------------------------------------------------
OptiView ReadOpti(const GameEntry& e) {
    OptiView o;
    string ini;
    Record m;
    const bool rec = e.hasRecord && LoadRecord(RecordPath(e.dir), m) && !m.legacy;
    o.have = rec && HostedRecord(m) && ReadAll(Join(e.dir, L"OptiScaler.ini"), ini);
    if (!o.have) {
        o.headline = rec ? L"WrongGPU's OptiScaler setup isn't installed in this game." : L"OptiScaler settings can be changed once WrongGPU is installed in this game.";
        return o;
    }
    o.iniText = ini;
    o.proxy = m.proxy; o.proxyReason = m.proxyReason; o.folder = e.dir;
    const wstring ver = !m.proxy.empty() && IsFile(Join(e.dir, m.proxy)) ? FileVersion(Join(e.dir, m.proxy)) : wstring();
    o.version = ver == L"unknown" ? wstring() : ver;
    o.headline = L"OptiScaler " + (o.version.empty() ? wstring() : o.version + L" ") + L"as " + (m.proxy.empty() ? wstring(L"its proxy DLL") : m.proxy) +
                 (m.proxyReason.empty() ? wstring() : L" (" + m.proxyReason + L")");
    for (auto& r : OptiTabRows(ini)) (r.locked ? o.locked : o.editable).push_back(r);
    for (auto& x : JournalOutside(ini, m.inikeys, L"OptiScaler.ini")) if (!IsOptiLocked(x.section, x.key)) o.outside.push_back(x);
    return o;
}

void DropOptiEdits(size_t keep) {  // the OptiScaler text fields from `keep` on: their windows destroyed
    for (size_t i = keep; i < v.edOpti.size(); ++i)
        if (v.edOpti[i].h) {
            if (GetFocus() == v.edOpti[i].h) SetFocus(v.hwnd);
            DestroyWindow(v.edOpti[i].h);
        }
    if (v.edOpti.size() > keep) v.edOpti.resize(keep);
}

// The game page's data from the game's own folder (its profile, OptiScaler's setup, what changed outside the launcher):
// small reads, but on the game's disk, which may be asleep or a slow network library. Any thread: a worker reads it before
// the page opens (OpenGame), after an operation (StartOp) and once a running game closed (timer 4), so the window never
// waits for that disk.
PageData ReadPageData(const GameEntry& e, const Profile& defaults) {
    PageData d;
    d.saved = e.hasRecord ? GameProfile(e.dir) : defaults;
    d.opti = ReadOpti(e);
    string wini, oini;
    ReadAll(Join(e.dir, L"wronggpu.ini"), wini);
    ReadAll(Join(e.dir, L"OptiScaler.ini"), oini);
    d.outside = e.hasRecord ? OutsideEdits(d.saved, wini, oini) : std::vector<OutsideEdit>{};
    if (const RrConfigGame* rg = e.hasRecord ? RrConfigGameFor(e.exe) : nullptr) {
        d.rrcfg = RrConfigLineState(g.ctx, *rg, &d.rrcfgFile, &d.rrcfgWhy);
        Record m;
        d.rrcfgFixed = LoadRecord(RecordPath(e.dir), m) && m.rrconfig.present;
    }
    d.have = true;
    return d;
}
// The same on a worker: false when it could not be read there (it is read where it is shown then, as before)
bool ReadPageDataSafe(const GameEntry& e, const Profile& defaults, PageData& d) {
    try { d = ReadPageData(e, defaults); return true; } catch (...) { d = PageData(); return false; }
}

// data: the page's files as a worker read them (else read here: a fixture, the ui-dump, a read that failed there)
void ShowEntryWith(const GameEntry& e, bool keepView, const PageData* data) {
    // the page shown while its files were read (pageReading): filled as a new page, without a second entrance
    const bool reading = v.pageReading && v.page == P_GAME;
    v.pageReading = false;
    const bool same = v.page == P_GAME && IEq(v.cur.dir, e.dir) && !reading;
    const bool fill = reading && IEq(v.cur.dir, e.dir);
    if (!IEq(v.rrcfgReveal, e.dir)) v.rrcfgReveal.clear();  // "Show me how" was for another game's page
    v.cur = e;
    if (!v.fixture) {
        // read here only without a worker's (the ui-dump; a live window's callers bring it: ReloadEntry, WM_APP_PAGE), and a
        // throw never leaves: the page keeps what it shows
        PageData d;
        if (data && data->have) d = *data;
        else if (!ReadPageDataSafe(e, gl.global.defaults, d)) {
            if (same) { d.saved = v.saved; d.opti = v.opti; d.outside = v.outside; d.rrcfg = v.rrcfg; d.rrcfgFile = v.rrcfgFile; d.rrcfgFixed = v.rrcfgFixed; d.rrcfgWhy = v.rrcfgWhy; }
            else d.saved = gl.global.defaults;
        }
        v.saved = d.saved;
        v.opti = d.opti;
        v.outside = d.outside;
        v.rrcfg = d.rrcfg; v.rrcfgFile = d.rrcfgFile; v.rrcfgFixed = d.rrcfgFixed; v.rrcfgWhy = d.rrcfgWhy;
        // "Fix it for me" chosen before the game's first start, and its settings folder is there now: applied (r2b #1)
        if (d.rrcfg >= 0 && !d.rrcfgFixed && d.saved.get("rr_config_fix") == "on") ApplyRrConfigWhenDue(e.dir, e.exe);
    }
    v.edit = v.saved;
    v.optiValues.clear();
    for (auto& r : v.opti.editable) v.optiValues.push_back(r.value);
    DropOptiEdits(same ? v.opti.editable.size() : 0);  // another game: its own fields (none left behind, no window leaked)
    if (!keepView || !same) { v.detailsOpen = v.lockedOpen = v.optiMore = false; v.scroll[P_GAME] = v.target[P_GAME] = 0; if (!fill) v.heroTall = -1; }
    // an Apply of this game that failed after its page was left: the values the player set come back as unsaved changes
    if (!same) {
        const auto kept = keptEdits.find(Lower(e.dir));
        if (kept != keptEdits.end()) {
            const View::PendingApply k = kept->second;
            keptEdits.erase(kept);
            for (auto& kv : k.edit.v) {
                const auto it = k.saved.v.find(kv.first);
                if (it == k.saved.v.end() || it->second != kv.second) v.edit.v[kv.first] = kv.second;
            }
            for (size_t i = 0; i < k.rows.size() && i < k.opti.size(); ++i) {
                if (k.opti[i] == k.rows[i].value) continue;
                for (size_t j = 0; j < v.opti.editable.size() && j < v.optiValues.size(); ++j)
                    if (v.opti.editable[j].section == k.rows[i].section && v.opti.editable[j].key == k.rows[i].key) v.optiValues[j] = k.opti[i];
            }
        }
    }
    // "Show me how" (r3 #6): Myself on the page, unsaved while the question's answer isn't saved (a protected folder: saving it
    // would ask Windows for permission; Apply does that on the player's click), so the steps show right away; once
    if (!v.rrcfgMyselfDir.empty() && IEq(v.rrcfgMyselfDir, e.dir)) {
        if (v.saved.get("rr_config_fix") == "ask") SetSetting(v.edit, "rr_config_fix", "off", nullptr);
        v.rrcfgMyselfDir.clear();
    }
    if (!keepView || !same) v.advOpen = AdvancedWanted();  // after the kept edits: an unsaved Advanced value shows (UI review #5)
    v.curIndex = -1;
    for (size_t i = 0; i < v.games.size(); ++i) if (IEq(v.games[i].dir, e.dir) && IEq(v.games[i].exe, e.exe)) v.curIndex = (int)i;
    if (v.page != P_GAME || (!same && !fill)) v.pageChangedAt = NowMs();
    v.page = P_GAME;
    if (!same) v.menu.open = false;
    else RefreshPageMenu();  // the same game read again: its open menu stays (ReloadEntry refreshes it once more after the edits)
    Invalidate();
}

void ShowEntry(const GameEntry& e, bool keepView) { ShowEntryWith(e, keepView, nullptr); }  // its files read here


// Unsaved changes put back over the values a page was just read again with: the profile keys the player changed from
// `base`, and the OptiScaler values changed from `rows` (matched by section and key)
void RestoreEdits(const Profile& edit, const Profile& base, const std::vector<OptiRow>& rows, const std::vector<string>& vals) {
    for (auto& kv : edit.v) {
        const auto it = base.v.find(kv.first);
        if (it == base.v.end() || it->second != kv.second) v.edit.v[kv.first] = kv.second;
    }
    for (size_t i = 0; i < rows.size() && i < vals.size(); ++i) {
        if (vals[i] == rows[i].value) continue;
        for (size_t j = 0; j < v.opti.editable.size() && j < v.optiValues.size(); ++j)
            if (v.opti.editable[j].section == rows[i].section && v.opti.editable[j].key == rows[i].key) v.optiValues[j] = vals[i];
    }
}

// The Library's entry of a game an operation runs on: by its exe (two entries can share a folder: an exe added next to the
// store's), else by its folder
GameEntry* EntryOf(const wstring& dir, const wstring& exe) {
    if (!exe.empty()) for (auto& x : v.games) if (IEq(x.exe, exe)) return &x;
    for (auto& x : v.games) if (IEq(x.dir, dir)) return &x;
    return nullptr;
}
wstring EntryExe(const wstring& dir, const wstring& exe, const GameEntry* cur) {
    const wstring theExe = !exe.empty() ? exe : cur ? cur->exe : Join(dir, L"?.exe");
    return theExe.empty() ? Join(dir, L"?.exe") : theExe;
}
// A repair of an older WrongGPU: an update too (PrepareOp asks so; the busy words and the result say so)
bool RepairUpdates(const OpRequest& r) {
    if (r.op != Op::Repair || !g.ctx.data) return false;
    const GameEntry* e = v.page == P_GAME && IEq(v.cur.dir, r.dir) ? &v.cur : EntryOf(r.dir, r.exe);
    return e && UpdateDue(*e, g.ctx);  // an older version, or another build of this one (rc5)
}
// The Library's lockups and its featured banner are named by their places in the list: once it changes, a press, a hover or
// a lift on one would land on another game. Only those are let go; a press on the toolbar, a tab, a Settings or game page
// control stays (its release still answers: the first click is never lost while a scan lands).
bool PlaceId(int id) { return id >= ID_TILE || (id >= ID_FEAT0 && id < ID_FEAT0 + 3); }
void ForgetPlaces() {
    if (PlaceId(v.down)) v.down = 0;
    if (PlaceId(v.hot)) v.hot = 0;
    for (auto it = v.anim.begin(); it != v.anim.end();) it = PlaceId(it->first) ? v.anim.erase(it) : std::next(it);
}
// show: the page of that game is shown again from it (after an operation); else only its entry (a scan's result, a game
// that started or closed: the values on the page are kept). fresh: the entry as a worker read it (status, record, last
// run); without it the game is read here (only where no worker did: a fixture, the ui-dump)
// page: the page's data read with `fresh` on that worker (else read here)
// A live window never reads a game on its own thread (it hashes the game's files, waits for its disk, and a throw there would
// end the process): what a worker could not read is read once more on another worker (ReadAgainLater, WM_APP_ENTRIES), the
// Library and the page keeping what they show until it lands. `again`: false for that second read's result (no third one).
void ReadAgainLater(const wstring& dir, const wstring& exe, const wstring& name, const wstring& source);
void ReloadEntry(const wstring& dir, const wstring& exe, bool show = true, const GameEntry* fresh = nullptr, const PageData* page = nullptr, bool again = true) {
    GameEntry* cur = EntryOf(dir, exe);
    const wstring theExe = EntryExe(dir, exe, cur);
    GameEntry e;
    if (fresh) e = *fresh;
    else if (v.live) {
        if (again) ReadAgainLater(dir, theExe, cur ? cur->name : L"", cur ? cur->source : L"");
        return;
    } else {
        try { e = EntryFor(g.ctx, theExe, cur ? cur->name : L"", cur ? cur->source : L""); }
        catch (...) { return; }  // the ui-dump: the entry stays as it was
    }
    const bool onPage = v.page == P_GAME && (IEq(v.cur.exe, theExe) || (exe.empty() && IEq(v.cur.dir, dir)));
    // a game that is gone and has nothing of WrongGPU left (its Uninstall just ran): it leaves the Library, as a scan has it.
    // Whether its program is there: as the worker read the entry (exeFound), never a file check on the window's thread
    const bool exeThere = v.live && !v.fixture ? e.exeFound : IsFile(theExe);
    if (!exeThere && !e.hasRecord && e.leftovers.empty()) {
        if (cur) {
            const int gone = (int)(cur - v.games.data());
            // the lockups' ids are their places, which move: the keyboard's (also under an open dialog) go with their games
            HoldPlaceFocus();
            v.games.erase(v.games.begin() + gone);
            RestorePlaceFocus();
            // the open page of another game keeps its place in the list (Back puts the focus on its lockup)
            if (v.curIndex > gone) --v.curIndex;
            else if (v.curIndex == gone) v.curIndex = -1;
            ForgetPlaces();
        }
        if (onPage) { v.page = P_LIBRARY; v.pageChangedAt = NowMs(); v.curIndex = -1; v.menu.open = false; }
        Invalidate();
        return;
    }
    if (cur) *cur = e; else v.games.push_back(e);
    if (e.rrConfigDue) ApplyRrConfigWhenDue(e.dir, e.exe);  // the Engine.ini choice waiting for the game's folder, which is there now (r2b #1)
    if (!onPage) return;
    if (!show) { v.cur = e; return; }
    // the page's files could not be read on the worker: its values stay as shown until a second read brings them
    if (v.live && !(page && page->have)) {
        v.cur = e;
        if (again) ReadAgainLater(dir, theExe, e.name, e.source);
        Invalidate();
        return;
    }
    // unsaved changes on the page stay (a Repair, Roll back or Update that finished on this game read its files again)
    const bool keep = Dirty();
    const Profile edit = v.edit, base = v.saved;
    const std::vector<OptiRow> rows = v.opti.editable;
    const std::vector<string> vals = v.optiValues;
    ShowEntryWith(e, true, page);
    if (keep) RestoreEdits(edit, base, rows, vals);
    RefreshPageMenu();  // the Output filter's check follows the unsaved choice put back
}
void ReadAgainLater(const wstring& dir, const wstring& exe, const wstring& name, const wstring& source) {
    if (!v.live || !v.hwnd) return;
    HWND hwnd = v.hwnd;
    const Profile defaults = gl.global.defaults;
    Worker([hwnd, dir, exe, name, source, defaults] {
        auto* m = new EntriesMsg;
        m->again = true;
        try {
            std::shared_lock<std::shared_mutex> opti(optiLock);
            EntriesMsg::Item it{dir, exe, EntryFor(g.ctx, exe, name, source), PageData()};
            ReadPageDataSafe(it.fresh, defaults, it.page);
            m->items.push_back(std::move(it));
        } catch (...) {  // not readable twice: the Library keeps the entry it had (the log says which game)
            try { LogLine(g.ctx, L"read ERROR: " + exe + L" could not be read again"); } catch (...) {}
        }
        if (!PostMessageW(hwnd, WM_APP_ENTRIES, 0, (LPARAM)m)) delete m;
    });
}

// ---- operations ------------------------------------------------------------------------------------------------------
void StartNextQueued();
void ScrollBy(float dy);
void StartOp(OpRequest r) {
    if (v.busy) return;
    StopRrBackground();  // the operation may replace the runtime the helper holds; the background run goes on after it
    v.busy = true;
    v.busyDir = r.dir;
    BlockShutdown(true);
    // what the plate and the bay say until the first step arrives (OpName is the CLI's verb, not words for people)
    static const std::pair<Op, const wchar_t*> kDoing[] = {
        {Op::Install, L"Installing\u2026"}, {Op::Repair, L"Repairing\u2026"}, {Op::Update, L"Updating\u2026"}, {Op::Uninstall, L"Uninstalling\u2026"},
        {Op::Rollback, L"Rolling back\u2026"}, {Op::Apply, L"Applying settings\u2026"}, {Op::Adopt, L"Adopting\u2026"},
        {Op::CleanLeftovers, L"Cleaning up\u2026"}, {Op::AdoptLeftovers, L"Adopting\u2026"}};
    v.step = L"Working\u2026";
    for (auto& d : kDoing) if (d.first == r.op) v.step = d.second;
    if (r.op == Op::Apply && r.optiReset) v.step = L"Resetting\u2026";  // More > Reset OptiScaler settings: named as it was chosen
    const bool rrSetup = r.op == Op::Repair && rrSetupNext;  // the RR row's Set up: named as it was chosen, not "Repairing"
    if (rrSetup) v.step = L"Setting up\u2026";
    else if (RepairUpdates(r)) v.step = L"Repairing and updating\u2026";  // as its question said (PrepareOp): an update too
    // a game that is gone (Orphaned): Remove files says what it does, as its button did (not "Uninstalling")
    const GameEntry* was = r.op == Op::Uninstall ? EntryOf(r.dir, r.exe) : nullptr;
    const bool gone = was && was->status == GameStatus::Orphaned;
    if (gone) v.step = L"Removing files\u2026";
    v.busyVerb = v.step;  // the Library's lockup keeps the verb while the steps change ("Updating… 62%")
    v.progress = 0;
    if (IEq(v.noticeDir, r.dir)) v.notice.clear();  // a failure's callout of another game stays until it is dismissed there
    {  // "Updating WrongGPU in Quarry Nine…" for Narrator (the result is a toast or a note, announced too; StartSaid)
        wstring who;
        for (auto& e : v.games) if (IEq(e.dir, r.dir)) who = e.name;
        if (who.empty() && IEq(v.cur.dir, r.dir)) who = v.cur.name;
        if (gone && !who.empty()) UiaNotify(L"Removing WrongGPU's files from " + who + L"'s old folder…");
        else if (!who.empty()) UiaNotify(StartSaid(r, v.step, rrSetup, who));
    }
    Invalidate();
    HWND hwnd = v.hwnd;
    // Update all: a game its elevated worker did already, with the queue's first game in a protected folder (one permission
    // prompt for them all): its result is taken as it is
    const auto preDone = batchedResults.find(Lower(r.dir));
    const bool pre = r.op == Op::Update && preDone != batchedResults.end();
    const OpResult preRes = pre ? preDone->second : OpResult();
    if (pre) batchedResults.erase(preDone);
    // ...and the first of them: the queue's other games in protected folders go into the same elevated worker. Whether a
    // folder is protected (a write test in it) is decided on the worker: no game folder is touched on the window's thread,
    // so a sleeping disk or a dropped network share never freezes the window before its busy state shows
    std::vector<OpRequest> mates;
    if (!pre && r.op == Op::Update && v.queueTotal > 1)
        for (auto& q : v.queue) if (q.op == Op::Update && !q.dir.empty()) mates.push_back(q);
    const GameEntry* cur = EntryOf(r.dir, r.exe);
    const wstring theExe = EntryExe(r.dir, r.exe, cur), name = cur ? cur->name : wstring(), source = cur ? cur->source : wstring();
    const Profile defaults = gl.global.defaults;  // the worker's own copy (Settings may change gl.global meanwhile)
    Worker([hwnd, r, pre, preRes, mates, theExe, name, source, defaults] {
        auto* done = new OpDoneMsg;
        done->dir = r.dir; done->exe = r.exe; done->op = r.op; done->req = r; done->name = name;
        bool elevate = pre;
        std::vector<OpRequest> batch;
        try {
            if (!pre && !r.dir.empty()) elevate = NeedsElevation(r.dir);
            if (elevate && !pre) for (auto& q : mates) if (NeedsElevation(q.dir)) batch.push_back(q);
        } catch (...) {}
        // the OptiScaler package: held alone while this operation may prepare it (the scan and the poll wait meanwhile),
        // else shared with them (optiLock)
        std::shared_lock<std::shared_mutex> shared(optiLock, std::defer_lock);
        std::unique_lock<std::shared_mutex> alone(optiLock, std::defer_lock);
        if (!elevate && OptiMayChange(r)) alone.lock(); else shared.lock();
        // everything that can throw stays inside: writing the elevated job (a full drive), the operation's log; a throw out of
        // a worker thread would end the process with no word (std::terminate). It ends as the operation's failure instead.
        try {
            RefreshPerGameEntry();  // the launcher installed or uninstalled meanwhile (a portable window stays open): this operation follows it
            if (pre) {
                done->res = preRes;
            } else if (elevate && r.noPrompt) {
                // the window's own apply (the Engine.ini choice once the game's settings folder is there, r2c): a protected
                // folder would raise a permission prompt nobody clicked for. Nothing is run; the card says "Ready to add"
                done->needsPermission = true;
                done->res.code = kOpNotRun;
                LogLine(g.ctx, L"rr_config_fix: " + r.dir + L" needs administrator permission: left for the player's click (Fix it now, Apply, Repair or Update)");
            } else if (elevate && !batch.empty()) {  // one permission prompt for the queue's games in protected folders; stop at a failure
                std::vector<OpRequest> all{r};
                all.insert(all.end(), batch.begin(), batch.end());
                const std::vector<OpResult> res = RunElevatedOps(all, hwnd, true);
                done->res = res[0];
                for (size_t i = 0; i < batch.size(); ++i) done->batched.push_back({batch[i].dir, res[i + 1]});
            } else if (elevate) {
                done->res = RunElevated(r, hwnd);
            } else {
                Ui ui;
                ui.step = [hwnd](int d, int t, const wstring& s) { auto* m = new StepMsg{d, t, s}; if (!PostMessageW(hwnd, WM_APP_STEP, 0, (LPARAM)m)) delete m; };
                ui.log = [](const wstring& s) { LogLine(g.ctx, s); };
                OpLogBegin();
                done->res = RunOp(g.ctx, r, ui);
            }
        } catch (const SetupError& e) {
            done->res = OpResult(); done->res.code = EXIT_FAILED; done->res.message = e.msg;
            LogLine(g.ctx, L"operation ERROR: " + e.msg);
        } catch (...) {
            done->res = OpResult(); done->res.code = EXIT_FAILED; done->res.message = L"internal error: the operation stopped unexpectedly";
            LogLine(g.ctx, L"operation ERROR: unexpected failure");
        }
        // an install, update or repair may have converted the Ray Reconstruction data: the RR games installed before it get
        // RR too, the ones in protected folders with permission (the result's warnings say what still could not be done)
        // (a game an earlier batch did: that batch's first game did this for them all)
        if (!pre && !done->res.code && (r.op == Op::Install || r.op == Op::Repair || r.op == Op::Update)) SyncRrSettings(hwnd, r.dir, done->res.warnings);
        if (!elevate) OpLogEnd();
        if (alone.owns_lock()) { alone.unlock(); shared.lock(); }  // written: the re-read below shares it again
        // the game read again here, not on the window's thread (it hashes every installed file), and its page's data with it
        try { done->fresh = EntryFor(g.ctx, theExe, name, source); done->haveEntry = true; } catch (...) {}
        if (done->haveEntry) ReadPageDataSafe(done->fresh, defaults, done->page);
        if (!PostMessageW(hwnd, WM_APP_OPDONE, 0, (LPARAM)done)) delete done;
    });
}

// A read before a question, on a worker while the window keeps painting and answering in a nested loop as a dialog's does:
// a game's files before an install (InspectAsync), every game's record before an uninstall question (a game on a network share
// that is offline holds each look for the share's timeout, never the window), the logs from the launcher's uninstall question.
// dir: the game (its lockup says verb), task: what it runs as (T_COLLECT); nothing else starts meanwhile (v.busy). work keeps
// its results in what it holds (shared). False: the window is closing (nothing after it is asked).
bool WaitForRead(const wstring& dir, const wstring& verb, int task, std::function<void()> work) {
    if (!v.live || !v.hwnd) { work(); return true; }
    auto done = std::make_shared<std::atomic<bool>>(false);
    v.busy = true; v.busyDir = dir; v.busyVerb = v.step = verb; v.progress = 0; v.task = task;
    inspecting = true;  // only reads: Windows' shutdown is not held up meanwhile (WritesNow)
    Invalidate();
    HWND hwnd = v.hwnd;
    Worker([work, done, hwnd] {
        try { work(); } catch (...) { LogLine(g.ctx, L"read ERROR: unexpected failure"); }
        *done = true;
        PostMessageW(hwnd, WM_NULL, 0, 0);  // the loop below looks again
    });
    MSG m;
    while (!*done) {
        const BOOL r = GetMessageW(&m, nullptr, 0, 0);
        if (r == 0) { PostQuitMessage((int)m.wParam); break; }  // the outer loop ends too (the questions after this are not asked)
        if (r < 0) break;
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    v.busy = false; v.busyDir.clear(); v.busyVerb.clear(); v.step.clear(); v.task = T_NONE;
    inspecting = false;
    Invalidate();
    return *done;
}

// A game's files read before an install, update or repair asks its questions (InspectGame parses the game's PE files and
// hashes OptiScaler's: seconds on a hard disk), off the window's thread (WaitForRead).
// `error`: the core's words when the game's files could not be read (a file another program holds, ...).
// `gone`: the game's program isn't there any more (looked at on the worker too: the window's thread never waits for the game's
// disk); nothing is inspected then.
// `prot` (Update all): set to whether the game's folder needs administrator rights (a write test there, on the worker too).
GameTarget InspectAsync(const GameEntry& e, wstring& error, bool& gone, bool* prot = nullptr) {
    gone = false;
    if (prot) *prot = false;
    if (!v.live || !v.hwnd) {
        if (!v.fixture && !IsFile(e.exe)) { gone = true; return GameTarget(); }
        if (prot && !e.dir.empty()) *prot = NeedsElevation(e.dir);
        try { return InspectGame(g.ctx, e.exe); } catch (const SetupError& x) { error = x.msg; return GameTarget(); }
    }
    auto out = std::make_shared<GameTarget>();
    auto err = std::make_shared<wstring>();
    auto missing = std::make_shared<std::atomic<bool>>(false);
    auto needs = std::make_shared<std::atomic<bool>>(false);
    const wstring exe = e.exe, dir = prot ? e.dir : wstring();
    const bool look = !v.fixture;
    if (!WaitForRead(e.dir, L"Preparing…", T_NONE, [out, err, missing, needs, exe, dir, look] {
            if (look && !IsFile(exe)) { *missing = true; return; }
            try { std::shared_lock<std::shared_mutex> opti(optiLock); *out = InspectGame(g.ctx, exe); }
            catch (const SetupError& x) { *err = x.msg.empty() ? wstring(L"the game's files could not be read") : x.msg; }
            catch (...) { *err = L"the game's files could not be read"; }
            if (!dir.empty()) try { *needs = NeedsElevation(dir); } catch (...) {}
        }))
        return GameTarget();  // the window is closing (the questions after this are not asked)
    error = *err; gone = *missing;
    if (prot) *prot = *needs;
    return *out;
}

void InformDetails(const wstring& text, int icon, const wstring& details);
// launcher.ini written (every Settings change, Add game, a folder, the first run's choice). It can fail (a sync client or
// antivirus holding the file, a full drive, a portable folder made read-only): then the change still holds until the
// launcher closes, and a note says so in words (the cause behind Show details), never a crash. False: not saved.
bool SaveSettings() {
    try {
        SaveGlobal(gl.paths, gl.global);
        return true;
    } catch (const SetupError& e) {
        LogLine(g.ctx, L"settings not saved: " + e.msg);
        const wstring low = Lower(e.msg);
        const bool full = low.find(L"not enough space") != wstring::npos || low.find(L"disk is full") != wstring::npos;
        InformDetails(wstring(L"Your settings couldn't be saved.\n\n") +
                          (full ? L"The drive is full. Free some space, then change the setting again. Until then the change holds only until you close the launcher."
                                : L"Windows didn't let WrongGPU write its settings: another program may have them open, or the folder is read-only. Try again in a "
                                  L"moment. Until then the change holds only until you close the launcher."),
                      kIconError, e.msg);
        return false;
    }
}

// The OptiScaler of an install (pkg): WrongGPU's edition of OptiScaler (optiedition.h: cached, or downloaded now with
// progress); else the player's own download (Settings), else one question and a picker. False: cancelled.
wstring editionPath;  // this session's checked edition archive
bool EnsureOptiPackage(const wstring& dir, wstring& pkg) {
    pkg.clear();
    if (!editionPath.empty() && IsFile(editionPath)) { pkg = editionPath; return true; }
    auto res = std::make_shared<EditionResult>();
    const wstring data = gl.paths.dataDir;
    HWND hwnd = v.hwnd;
    const bool live = v.live && v.hwnd;
    if (!WaitForRead(dir, L"Downloading OptiScaler\u2026", T_NONE, [res, data, hwnd, live] {
            int last = -1;
            *res = GetEdition(g.ctx, data, [hwnd, live, &last](unsigned long long d, unsigned long long t) {
                const int pct = t ? (int)(d * 100 / t) : 0;
                if (live && pct != last) {
                    last = pct;
                    auto* m = new StepMsg{pct, 100, L"Downloading OptiScaler"};
                    if (!PostMessageW(hwnd, WM_APP_STEP, 0, (LPARAM)m)) delete m;
                }
                return true;
            });
        }))
        return false;  // the window is closing
    if (!res->path.empty()) { editionPath = res->path; pkg = editionPath; return true; }
    LogLine(g.ctx, L"OptiScaler: WrongGPU's edition not available (" + res->detail + L"); the player's own OptiScaler download is used");
    if (!gl.global.optiPackage.empty() && Exists(gl.global.optiPackage)) { pkg = gl.global.optiPackage; return true; }
    // the file (.7z, .zip) or the unpacked folder: one question, then one picker
    const int a = RunDialog(2, L"Choose your OptiScaler download\n\n" + res->error + L" You can choose an OptiScaler download yourself: the file you downloaded, or the "
                               L"folder you unpacked it to. WrongGPU is tested with OptiScaler " + wstring(kOptiTested) + L". Games whose Ray Reconstruction needs "
                               L"the WrongGPU edition of OptiScaler get it once the download works again.",
                            {{L"Cancel", 0}, {L"Choose folder\u2026", 2}, {L"Choose file\u2026", 1, true}}, 0);
    if (a == 0) return false;
    wstring p;
    if (!Browse(a == 1 ? 3 : 4, p)) return false;
    gl.global.optiPackage = p;
    SaveSettings();  // the choice holds for this session either way
    pkg = p;
    return true;
}

void InformDetails(const wstring& text, int icon, const wstring& details);
wstring PlainCause(const wstring& raw, int code);  // a core message as a cause in words (defined with the result words)
void StartScan();
// The request of an install / update / repair / uninstall / roll back with the player's answers; false: cancelled (or it
// could not be prepared: said in a note, and *failed set, so Update all goes on with its other games).
// `gone` (Update all): a game whose program isn't there any more returns false with it set, without a note (the others go on).
bool PrepareOp(const GameEntry& e, Op op, OpRequest& r, bool* failed, bool* gone = nullptr) {
    if (failed) *failed = false;
    if (gone) *gone = false;
    wstring pkg;
    if (op != Op::Uninstall && op != Op::Rollback && !EnsureOptiPackage(e.dir, pkg)) return false;
    r = OpRequest(); r.op = op; r.exe = e.exe; r.dir = e.dir; r.optiPackage = pkg; r.noScan = g.noScan;
    r.hasProfile = true; r.profile = gl.global.defaults;
    if (op == Op::Install || op == Op::Update || op == Op::Repair) {
        wstring readError;
        bool missing = false, prot = false;
        const GameTarget t = InspectAsync(e, readError, missing, gone ? &prot : nullptr);
        if (gone && !missing && readError.empty()) queueProtected[Lower(e.dir)] = prot;  // Update all's order (UpdateAll)
        // the game's program gone since the last scan (moved, uninstalled): said as it is, and the Library looks again, never
        // an "Install anyway?" for a game that isn't there followed by a failure that asks for a rescan
        if (missing) {
            if (gone) { *gone = true; return false; }
            Inform(e.name + L" isn't in its folder any more.\n\nThe game may have moved or been uninstalled. The Library looks for it again.", 1);
            StartScan();
            if (failed) *failed = true;
            return false;
        }
        if (!readError.empty()) {  // the game's files could not be read: a note in words, the core's text behind Show details
            LogLine(g.ctx, wstring(L"prepare ") + OpName(op) + L" " + e.dir + L": " + readError);
            InformDetails(OpFailTitle(op, e.name) + L"\n\n" + PlainCause(readError, 0) + L" Nothing was changed.", kIconError, readError);
            if (failed) *failed = true;
            return false;
        }
        if (t.exe.empty() && t.name.empty()) return false;  // the window is closing
        if (!t.validated && !t.installed && !Confirm(L"Install WrongGPU in " + t.name + L" anyway?\n\n" + t.name +
                                                         L" hasn't been tested with WrongGPU yet. It may not work, or the game may use AMD FSR instead.", 1,
                                                     L"Install anyway", L"Cancel"))
            return false;
        r.experimental = true;
        if (t.conflicts.modsFound()) {
            wstring list; for (auto& m : t.conflicts.mods) list += (list.empty() ? L"" : L", ") + m;
            if (!Confirm(L"Switch to WrongGPU's OptiScaler setup?\n\nThis game already has " + list + L". Your files are backed up and put back exactly as they were if you uninstall WrongGPU.",
                         0, L"Switch", L"Cancel"))
                return false;
            r.moveMods = true;
        }
        if (t.conflicts.oldFound()) {
            if (!Confirm(L"Remove the older WrongGPU install first?\n\nAn older WrongGPU (DLSS 4.5 AMD) install is in this game. The game's original files are put back exactly as they were.", 0,
                         L"Remove it first", L"Cancel"))
                return false;
            r.removeOld = true;
        }
#if !WRONGGPU_PRESET_K
        if (t.presetK && !Confirm(PresetKText(t.name, g.ctx.data->version), 0, L"Switch to Model M", L"Not now")) return false;
#endif
    }
    if (op == Op::Uninstall) {
        // a game that is gone (Orphaned): Remove files, said as what it does (there is no game to get its files back)
        const bool orphaned = e.status == GameStatus::Orphaned;
        const wstring name = e.name, dir = e.dir;  // kept: a scan may replace the Library's entries while the look below runs
        // the last game with WrongGPU and no launcher that keeps the shared data: the converted model data goes with it
        // (DoRestore). Looked at on a worker ("Preparing..."): it reads every game's record and the drives they are on
        auto data = std::make_shared<int>(0);
        if (!WaitForRead(dir, L"Preparing\u2026", T_NONE, [data, dir] {
                if (OtherGameWithWrongGPU(dir).empty() && !LauncherKeepsSharedData()) *data = IsFile(RrDataFile()) ? 2 : 1;
            }))
            return false;
        if (!Confirm(UninstallGameText(name, orphaned, *data), 0, orphaned ? L"Remove files" : L"Uninstall", L"Cancel", true)) return false;
    }
    // an older launcher never puts its older files over a newer WrongGPU (a download of an older release, opened by hand)
    if (op != Op::Uninstall && op != Op::Rollback && e.hasRecord && !e.installedVersion.empty() && VersionLess(g.ctx.data->version, e.installedVersion)) {
        Inform(e.name + L" has WrongGPU " + e.installedVersion + L", newer than this launcher (" + g.ctx.data->version + L").\n\nOpen the newer WrongGPU from Start to "
               L"change it. Nothing was changed.", 1);
        if (failed) *failed = true;
        return false;
    }
    // Repair puts this release's files back: on an older WrongGPU that is an update as well, and the player says so first (the
    // RR row's Set up, a repair too, asks in its own words)
    if (op == Op::Repair && UpdateDue(e, g.ctx)) {  // also another build of this version (rc5)
        if (!Confirm(RepairUpdateText(e.name, InstalledLabel(e), g.ctx.data->version, rrSetupNext), 0, rrSetupNext ? L"Set up" : L"Repair and update", L"Cancel")) return false;
    }
    if (op == Op::Rollback) { const wstring ver = e.previous; if (!Confirm(L"Roll " + e.name + L" back to WrongGPU " + ver + L"?", 0, L"Roll back", L"Cancel")) return false; }
    return true;
}
void SayResultAs(int kind, const wstring& dir);
void OnInstall(const GameEntry& e, Op op) {
    if (v.busy || v.listCached) return;  // listCached: the last session's list, its entries not read yet (its buttons are off)
    const bool gone = op == Op::Uninstall && e.status == GameStatus::Orphaned;  // Remove files: its result and failure in its words
    OpRequest r;
    if (PrepareOp(e, op, r, nullptr)) { StartOp(r); if (gone && v.busy) SayResultAs(RES_GONE, r.dir); }
}
void OnLeftovers(const GameEntry& e, bool clean) {
    if (v.listCached) return;  // the last session's list: the leftovers are not read yet (its buttons are off)
    wstring names;
    for (auto& f : e.leftovers) names += (names.empty() ? L"" : L"\n") + f.rel + L" (WrongGPU " + f.version + L")";
    for (;;) {  // Show details: the files' names, then the question again
        const int a = RunDialog(0, LeftoversText(e.name, e.leftovers, clean), LeftoversButtons(clean), 0);
        if (a == 1) break;
        if (a != 2 || !v.live) return;
        Inform(L"Details\n\n" + names, 2);
    }
    OpRequest r; r.op = clean ? Op::CleanLeftovers : Op::AdoptLeftovers; r.exe = e.exe; r.dir = e.dir;
    StartOp(r);
}
// An install the launcher does not manage yet (WrongGPU setup 0.5, no profile.ini, its own uninstall entry): adopted,
// its settings and files kept
void OnAdopt(const GameEntry& e) {
    if (v.busy || v.listCached) return;
    if (!Confirm(L"Adopt " + e.name + L"?\n\nWrongGPU " + e.installedVersion + L" was set up outside the launcher. Adopting keeps its files and settings; then the launcher "
                 L"updates, repairs and uninstalls it.", 0, L"Adopt", L"Cancel"))
        return;
    OpRequest r; r.op = Op::Adopt; r.exe = e.exe; r.dir = e.dir;
    StartOp(r);
}
// Play: the game starts, and its lockup and page show it running at once (Play, the settings and the actions wait), so a
// second press (a double click) does not try to start it again; timer 4 looks every 3 s whether a running game closed.
void InformDetails(const wstring& text, int icon, const wstring& details);
wstring PlainCause(const wstring& raw, int code);  // a core message as a cause in words (defined with the result words)
// ui_play.cpp: the running games (the running-game watch, timer 4; the games Play started, which a store may start again:
// Launching; a game shown running at Play's press, and as it was again when Windows didn't start it: PlayEnded)
void WatchRunning();
extern std::map<wstring, ULONGLONG> launchedAt;
bool Launching(const wstring& exe);
void KeepLaunching();
void ShowRunning(const wstring& exe);
void PlayEnded(const GameEntry& was, bool ok, bool running, bool gone, const wstring& why);
struct PlayMsg { GameEntry was; bool ok = false, running = false, gone = false; wstring why; };
void Play(const GameEntry& e) {
    if (PlayBlocked(e)) return;  // the game running, an operation in it or one that changes every game (not other games' work)
    StopRrBackground();  // the background re-prepare never runs beside a game (it goes on at the next scan's or operation's end)
    if (!RrReadyBeforePlay(e.dir, e.exe)) return;  // 0.6.0: RR's kernels first (ui_rrprep.cpp)
    // shown running at once (its lockup and page lock: a second press, a double click, does not start it again; read again on
    // a worker once it closes: timer 4), and its store may start it again in a moment (Launching)
    launchedAt[Lower(e.exe)] = GetTickCount64();
    played[Lower(e.exe)] = PlayedGame{GetTickCount64(), false};  // rc6: starting until seen running (WatchPlayed)
    ShowRunning(e.exe);
    WatchRunning();
    clickGuardAt = GetTickCount64();  // starting it can take a moment: a double click's second click does not press the pill again
    Invalidate();
    // Windows asked on a worker: the exe looked at, the running processes listed and the game started there, so a game on a
    // sleeping disk or a dropped network share never freezes the window (WM_APP_PLAYED says how it went)
    if (!v.live || !v.hwnd) {
        wstring why;
        const bool ok = PlayGame(e.exe, &why, nullptr, &g.ctx);  // through Steam for a Steam game, through Epic for an Epic one (logged)
        const bool running = !ok && GameRunning(e.exe);
        PlayEnded(e, ok, running, !ok && !running && !IsFile(e.exe), why);
        return;
    }
    HWND hwnd = v.hwnd;
    Worker([hwnd, e] {
        auto* m = new PlayMsg;
        m->was = e;
        // the shell (a game that asks for administrator rights starts through ShellExecuteEx) wants COM on its thread
        const HRESULT co = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        try {
            m->ok = PlayGame(e.exe, &m->why, nullptr, &g.ctx);  // through Steam for a Steam game, through Epic for an Epic one (logged)
            m->running = !m->ok && GameRunning(e.exe);
            m->gone = !m->ok && !m->running && !IsFile(e.exe);
        } catch (...) { m->ok = false; m->why = L"the game could not be started"; }
        if (SUCCEEDED(co)) CoUninitialize();
        if (!PostMessageW(hwnd, WM_APP_PLAYED, 0, (LPARAM)m)) delete m;
    });
}
void NoteLater(const wstring& text, int icon, const wstring& details);
void UpdateAll() {
    if (v.busy) return;
    v.queue.clear();
    v.queueDirs.clear();
    // a copy: PrepareOp's questions run modal loops, and a scan's result arriving in one replaces v.games
    const std::vector<GameEntry> games = v.games;
    bool gone = false;
    queueSkipped.clear();
    batchedResults.clear();
    queueProtected.clear();
    // every game it will update waits for it from the start (PlayBlocked: Play, the page's settings), not only those prepared
    // so far: each read before its questions takes seconds with the window answering, and a game started meanwhile would be
    // refused when its turn came. Free again once the queue is made (the queued ones wait in it), or when nothing starts.
    v.queueComing.clear();
    for (const GameEntry& e : games) if (e.status == GameStatus::UpdateAvailable) v.queueComing.push_back(e.dir);
    struct Coming { ~Coming() { v.queueComing.clear(); Invalidate(); } } coming;
    for (const GameEntry& e : games) {
        if (e.status != GameStatus::UpdateAvailable) continue;
        OpRequest r;
        bool failed = false, missing = false;
        if (!PrepareOp(e, Op::Update, r, &failed, &missing)) {
            // gone since the scan: not updated (a scan drops it), the others still are; said at the end (QueueSkippedWords)
            if (missing) { gone = true; queueSkipped.push_back({e.name, true}); continue; }
            // its files could not be read (its note said so): this game is left out, the others still are updated
            if (failed) { queueSkipped.push_back({e.name, false}); continue; }
            // cancelled: nothing starts, and nothing prepared so far is left to start later
            v.queue.clear(); v.queueDirs.clear(); v.queueTotal = v.queueDone = 0;
            queueSkipped.clear();
            Invalidate();
            return;
        }
        v.queue.push_back(r);
        v.queueDirs.push_back(e.dir);
    }
    // the games in protected folders last, together: the first of them takes the others into its elevated worker (one
    // permission prompt for the action, not one per game: StartOp), and the stop at a failure keeps the queue's order
    // (each folder's protection was found as its game was read, on a worker: PrepareOp)
    std::stable_partition(v.queue.begin(), v.queue.end(), [](const OpRequest& q) {
        const auto it = queueProtected.find(Lower(q.dir));
        return q.dir.empty() || it == queueProtected.end() || !it->second;
    });
    queueProtected.clear();
    v.queueDirs.clear();
    for (auto& q : v.queue) v.queueDirs.push_back(q.dir);
    v.queueTotal = (int)v.queue.size(); v.queueDone = 0;
    // none left to update: what happened to the ones it promised (a game not readable had its note already)
    if (v.queue.empty() && std::any_of(queueSkipped.begin(), queueSkipped.end(), [](const std::pair<wstring, bool>& s) { return s.second; })) NoteLater(QueueSkippedWords(), 1, L"");
    if (v.queue.empty()) queueSkipped.clear();
    StartNextQueued();
    if (gone) StartScan();
}
void StartNextQueued() {
    if (v.queue.empty()) { v.queueTotal = v.queueDone = 0; Invalidate(); return; }
    OpRequest r = v.queue.front();
    v.queue.erase(v.queue.begin());
    StartOp(r);
}

// Apply: the profile shown and the OptiScaler values changed, in one operation. While it writes, the values count as the
// game's (the capsule goes, leaving the page is not asked about again); if it fails they come back as unsaved changes
// (WM_APP_OPDONE), never lost.
void OnApply() {
    if (PageWritesWait()) return;  // one rule for every write of the page (its Apply says why)
    OpRequest r; r.op = Op::Apply; r.dir = v.cur.dir; r.hasProfile = true; r.profile = v.edit;
    for (size_t i = 0; i < v.opti.editable.size() && i < v.optiValues.size(); ++i)
        if (v.optiValues[i] != v.opti.editable[i].value) r.optiEdits.push_back({v.opti.editable[i].section + "." + v.opti.editable[i].key, v.optiValues[i]});
    for (auto& e : r.optiEdits) {  // the values typed are checked before anything is written
        string sec, key, normal; wstring why;
        const OptiKeyDef* d = SplitOptiName(e.first, sec, key) ? FindOptiKey(sec, key) : nullptr;
        if (d && !ValidateOptiValue(*d, e.second, &normal, &why)) {  // in the row's own words (the core's reason names the ini key)
            const wstring label = d->label;
            const wstring need = d->kind == OptiKind::Number ? L"needs a number, such as 1.0, or Auto."
                               : d->kind == OptiKind::Int ? L"needs a whole number, such as 1, or Auto."
                               : d->kind == OptiKind::Bool ? L"needs On, Off or Auto." : L"needs a key from its list, or Auto.";
            Inform(label + L" " + need, 1);
            FocusOptiField(e.first, L"N" + need.substr(1));  // the field that is wrong takes the keys, its value selected, marked invalid (ui_uia.cpp)
            return;
        }
        if (d) e.second = normal;
    }
    v.pendingApply = View::PendingApply{v.cur.dir, v.edit, v.saved, v.opti.editable, v.optiValues};
    v.saved = v.edit;
    for (size_t i = 0; i < v.opti.editable.size() && i < v.optiValues.size(); ++i) v.opti.editable[i].value = v.optiValues[i];
    clickGuardAt = GetTickCount64();  // the capsule closes at once: a double click's second click does not press the row under it
    StartOp(r);
}
// The game's profile written as `p` (the outside-change note's two buttons: the file's values kept, or WrongGPU's put
// back), and nothing else: the page's other unsaved changes stay unsaved, as they were (an Upscaler value, an OptiScaler
// field). A value the player has changed on the page stays theirs; the others show what was written. A failure brings
// the page back as it was (WM_APP_OPDONE).
void WriteProfileOnly(const Profile& p) { WriteProfileOnly(p, false); }
void WriteProfileOnly(const Profile& p, bool noPrompt) {
    if (PageWritesWait()) return;
    OpRequest r; r.op = Op::Apply; r.dir = v.cur.dir; r.hasProfile = true; r.profile = p; r.noPrompt = noPrompt;
    v.pendingApply = View::PendingApply{v.cur.dir, v.edit, v.saved, v.opti.editable, v.optiValues};
    for (auto& kv : p.v)
        if (v.edit.get(kv.first) == v.saved.get(kv.first)) v.edit.v[kv.first] = kv.second;  // not changed on the page
    v.saved = p;
    StartOp(r);
}
void OnOptiFix(bool keep) {
    if (PageWritesWait()) return;
    OpRequest r; r.op = Op::Apply; r.dir = v.cur.dir; r.optiKeep = keep; r.optiOnly = true;
    if (!keep) for (auto& x : v.opti.outside) r.optiEdits.push_back({x.section + "." + x.key, x.ours});
    StartOp(r);
}

// ---- first run, scan, health ----------------------------------------------------------------------------------------
void StartScan() {
    // one is running already, with the settings of when it started: one more once it ends (a game or folder just added,
    // the folder list just cleared)
    if (v.scanning) { v.rescanPending = true; return; }
    v.scanning = true;
    doneSinceScan.clear();
    Invalidate();
    HWND hwnd = v.hwnd;
    GlobalSettings gs = gl.global;
    // nothing listed yet (start-up): the games with WrongGPU are shown as soon as they are read (WPARAM 1), with the
    // featured banner and Play, while the store libraries are still read (seconds); a rescan keeps its list until the end
    const bool early = v.games.empty();
    Worker([hwnd, gs, early] {
        std::function<void(const std::vector<GameEntry>&)> first;
        if (early)
            first = [hwnd](const std::vector<GameEntry>& managed) {
                auto* p = new ScanMsg{managed};
                if (!PostMessageW(hwnd, WM_APP_SCAN, 1, (LPARAM)p)) delete p;
            };
        ScanMsg* m = nullptr;  // null: the scan failed (the Library keeps its list and says so: WM_APP_SCAN)
        try { std::shared_lock<std::shared_mutex> opti(optiLock); m = new ScanMsg{ScanGames(g.ctx, gs, g.noScan, first)}; }
        catch (const SetupError& e) { LogLine(g.ctx, L"scan ERROR: " + e.msg); }
        catch (...) { LogLine(g.ctx, L"scan ERROR: unexpected failure"); }
        if (!PostMessageW(hwnd, WM_APP_SCAN, 0, (LPARAM)m)) delete m;
    });
}

void StartFirstRun() {
    if (firstRunBusy) return;
    firstRunBusy = true;
    if (!v.first.open) {
        v.first.at = NowMs();  // its entrance (not again when it checks once more)
        v.focus = 0;  // the sheet itself has the focus while it checks (UI Automation: the dialog); then Continue (WM_APP_FIRSTRUN)
    }
    v.first.open = true; v.first.busy = true; v.first.lines.clear();
    // it scans the games too: the Library behind the sheet shows its skeleton meanwhile, not "No games found yet" (a scan
    // asked for meanwhile runs once it ends: rescanPending)
    v.scanning = true;
    Invalidate();
    HWND hwnd = v.hwnd;
    const GlobalSettings gs = gl.global;
    const wstring dll = g.userDll;
    const bool dump = !v.live;
    Worker([hwnd, gs, dll, dump] { FirstRunChecks(hwnd, gs, dll, dump); });  // ui_checks.cpp
}

void SetGpu(const GpuInfo& gpu) {
    wstring n = gpu.name;
    for (const wchar_t* p : {L"AMD Radeon ", L"AMD "}) if (n.rfind(p, 0) == 0) { n = n.substr(wcslen(p)); break; }
    v.gpuName = n;
    v.executorLine = n.empty() ? L"No graphics card found" : n + (gpu.rdna4 ? L" \u00b7 Direct3D 12" : gpu.amd ? L" \u00b7 HIP" : L" \u00b7 Not supported");
}
void SetHealth(bool model, const GpuInfo& gpu) { v.modelReady = model; SetGpu(gpu); }
// The model data's state for the toolbar and Settings (any thread): ready or why not, the DLSS file it came from, the HIP
// runtime line the first run's probe left.
static std::atomic<unsigned> healthReads{0};  // ReadModelState's reads in the order they began (HealthMsg::seq)
void ReadModelState(HealthMsg& m) {
    m.seq = ++healthReads;
    m.model = ExistingModelValid(g.ctx, &m.why);
    if (m.model) m.source = ReadModelSource(g.ctx);
    // a portable window that uses the shared data notes itself there, so a game's own uninstall leaves it (LauncherKeepsSharedData)
    if (v.live && gl.paths.portable && (m.model || IsFile(RrDataFile()))) NotePortableLauncher(gl.paths.exe);
    string raw;
    if (ReadAll(Join(AppDataPath(), L"hip-runtime.txt"), raw)) m.hip = FromUtf8(raw);  // only read: the folder is not made again
    while (!m.hip.empty() && (m.hip.back() == L'\n' || m.hip.back() == L'\r')) m.hip.pop_back();
    m.rr = RrDataValid(g.ctx, &m.rrWhy);  // the Ray Reconstruction data (it hashes the file too)
    if (m.rr) m.rrSource = RrDataSource();
}
// The model data's state shown: from the newest read only. Reads run on workers and end in any order (a conversion's own read,
// the one after every operation): one that began before the state shown now (it may have caught the file mid-swap) is
// dropped, never shown over a newer one.
static unsigned healthTaken = 0;
void TakeModelState(const HealthMsg& m) {
    if (m.seq && m.seq < healthTaken) return;
    if (m.seq) healthTaken = m.seq;
    v.modelReady = m.model; v.modelWhy = m.why; v.modelSource = m.source; v.hipLine = m.hip;
    v.rrReady = m.rr; v.rrWhy = m.rrWhy; v.rrSource = m.rrSource;
}
void RefreshModelNow() { HealthMsg m; ReadModelState(m); TakeModelState(m); }  // ui-dump only (no window on screen): a live window reads it on a worker
// At start-up, until StartHealth's check ends (it reads and hashes both model files: seconds on a hard disk): the files on
// disk as they were left, so the toolbar, Settings and the RR rows show the last known state at once, never "not
// converted" with an offer to convert valid data again. The check then confirms it, or says what is wrong.
void TakeModelStateOnDisk() {
    HealthMsg m;
    m.model = IsFile(ModelFile(g.ctx));
    if (m.model) m.source = ReadModelSource(g.ctx);
    string raw;
    if (ReadAll(Join(AppDataPath(), L"hip-runtime.txt"), raw)) m.hip = FromUtf8(raw);
    while (!m.hip.empty() && (m.hip.back() == L'\n' || m.hip.back() == L'\r')) m.hip.pop_back();
    m.rr = g.ctx.data && g.ctx.data->fMapId && IsFile(RrDataFile());
    if (m.rr) m.rrSource = RrDataSource();
    TakeModelState(m);
}
// Off the UI thread: at start-up (with the graphics card), and after every operation (an install can convert the model,
// the last uninstall removes it), so the toolbar's status line never stays behind.
// One check at a time (it hashes both data files: Update all would start one per game): asked while one runs, it runs once
// more when that one ends (WM_APP_HEALTH), with the graphics card if any ask wanted it.
static bool healthRunning = false, healthAgain = false, healthAgainGpu = false;
void StartHealth(bool gpu) {
    if (healthRunning) { healthAgain = true; healthAgainGpu = healthAgainGpu || gpu; return; }
    healthRunning = true;
    HWND hwnd = v.hwnd;
    Worker([hwnd, gpu] {
        auto* m = new HealthMsg;
        try { ReadModelState(*m); }  // a throw (out of memory hashing the file): not ready, why in words, never the process
        catch (...) { *m = HealthMsg(); m->why = L"the model data could not be read"; m->rrWhy = L"the Ray Reconstruction data could not be read"; }
        if (gpu) { try { m->gpu = DetectGpu(); } catch (...) { m->gpu = GpuInfo(); } m->gpuDone = true; }
        if (!PostMessageW(hwnd, WM_APP_HEALTH, 0, (LPARAM)m)) delete m;
    });
}
void HealthEnded() {
    healthRunning = false;
    if (!healthAgain) return;
    const bool gpu = healthAgainGpu;
    healthAgain = healthAgainGpu = false;
    StartHealth(gpu);
}

// ---- background tasks ------------------------------------------------------------------------------------------------
// A long job that is not a game operation, on a worker thread (the window keeps painting; v.busy: no operation starts
// meanwhile, the window does not close half way): `work` runs there and returns what the window then does (WM_APP_TASK).
void RunTask(int kind, std::function<std::function<void()>()> work) {
    v.busy = true; v.busyDir.clear(); v.step.clear(); v.busyVerb.clear(); v.task = kind;  // never an earlier operation's verb (BusyDoneWords)
    if (kind != T_COLLECT) BlockShutdown(true);  // collecting logs only reads
    // said as it starts (its button turns "Converting..." and keeps the focus): Narrator hears the job, then its result
    // (collecting logs has its own toast; the first run its sheet)
    UiaNotify(kind == T_MODEL ? L"Converting the Model M data\u2026" : kind == T_RRDATA ? L"Converting the Ray Reconstruction data\u2026"
              : kind == T_SELFINSTALL ? L"Installing WrongGPU for this user\u2026" : kind == T_SELFUNINSTALL ? L"Uninstalling the launcher\u2026" : L"");
    Invalidate();
    HWND hwnd = v.hwnd;
    // a failure the task did not expect ends as a note titled with what it was doing (the window stays, v.busy clears), never
    // as a crash half way
    const wstring title = kind == T_MODEL ? L"Model M couldn't be converted."
                          : kind == T_RRDATA ? L"The Ray Reconstruction data couldn't be converted."
                          : kind == T_SELFINSTALL || (kind == T_FIRSTRUN && v.first.installOffer && v.first.choice == 1) ? L"WrongGPU couldn't be installed for this user."
                          : kind == T_FIRSTRUN ? L"WrongGPU couldn't be set up."  // portable, or no choice: the model and the adoptions only
                          : kind == T_SELFUNINSTALL ? L"The launcher couldn't be uninstalled."
                          : kind == T_COLLECT ? L"The logs couldn't be collected." : L"WrongGPU couldn't finish that task.";
    // the launcher's own tasks say their cause as the launcher's (its folder, its entry; the logs file), never the game's folder
    const int cause = kind == T_COLLECT ? kCauseLogs : kind == T_SELFUNINSTALL ? kCauseLauncherUninstall
                      : kind == T_SELFINSTALL || (kind == T_FIRSTRUN && v.first.installOffer && v.first.choice == 1) ? kCauseLauncher : -1;
    Worker([hwnd, work, title, cause] {
        std::function<void()> done;
        try { done = work(); }
        catch (const SetupError& e) {
            const wstring msg = e.msg;
            LogLine(g.ctx, L"task ERROR: " + msg);
            done = [msg, title, cause] { InformDetails(title + L"\n\n" + (cause < 0 ? PlainCause(msg, 0) : PlainTaskCause(msg, cause)), kIconError, msg); };
        }
        catch (...) {
            LogLine(g.ctx, L"task ERROR: unexpected failure");
            done = [title] { Inform(title + L"\n\nSomething went wrong. Try again; if it keeps happening, collect the logs and report it.", kIconError); };
        }
        auto* m = new TaskMsg{done};
        if (!PostMessageW(hwnd, WM_APP_TASK, 0, (LPARAM)m)) delete m;
    });
}

// The Ray Reconstruction data converted again (Settings' Convert again / Choose a DLSS file..., a game page's Convert), off
// the window's thread: `dll` first, else the RR model files of the launcher's games and the store libraries. The data in
// use stays until the new one is checked; every installed RR game's settings file follows (ConvertRrData), and the
// Library is read again so the pages show it.
bool Has(const wstring& text, const wchar_t* part);
wstring PlainWarning(const wstring& w);
// RrGameRunningSaid, ModelGameRunningSaid: ui_checks.cpp
void ConvertRr(const wstring& dll) {
    if (v.busy) return;  // an install may be writing a game's RR settings file
    if (RrGameRunningSaid()) return;
    const bool noScan = g.noScan, had = v.rrReady;  // data in use before it (a failure's note says it stays only then)
    HWND hwnd = v.hwnd;
    RunTask(T_RRDATA, [dll, noScan, hwnd, had]() -> std::function<void()> {
        const std::vector<StoreGame> libs = noScan ? std::vector<StoreGame>{} : ScanLibraries();
        wstring why;
        std::vector<RrSettingsFailure> notWritten;
        std::vector<wstring> warnings;
        const bool ok = ConvertRrData(g.ctx, dll, libs, &why, &notWritten);
        // the RR games in protected folders: their settings files with permission (one prompt), else said
        if (ok) SyncRrSettings(hwnd, L"", warnings, &notWritten);
        HealthMsg hm;
        ReadModelState(hm);
        return [ok, why, warnings, hm, had, chose = !dll.empty()] {
            TakeModelState(hm);
            StartScan();  // the games' RR state (it turns on where their profile wants it)
            if (ok) {  // as Model M's: where it came from, and that the games that use it have it (ModelConvertedWords)
                const wstring game = v.rrSource.empty() ? wstring() : ModelSourceGame(v.rrSource);
                const wstring from = !game.empty() ? game + L"'s DLSS file" : chose ? wstring(L"the file you chose") : wstring(L"a DLSS file on this PC");
                const wstring head = L"The Ray Reconstruction data is converted from " + from + L".";
                if (warnings.empty()) {
                    const bool any = std::any_of(v.games.begin(), v.games.end(), [](const GameEntry& x) { return x.rrInstalled && x.rrUses && !x.rrHeldOff; });  // rc5
                    ShowToast(head + (any ? L" It's on in your games that use it." : L""), 1);
                    return;
                }
                // nothing fails silently: the games where it stays off, in words
                wstring body, raw;
                for (auto& w : warnings) {
                    const wstring p = PlainWarning(w);
                    if (!p.empty() && body.find(p) == wstring::npos) body += (body.empty() ? L"" : L" ") + p;
                    raw += (raw.empty() ? L"" : L"\n\n") + w;
                }
                InformDetails(head + L"\n\n" + body, 2, raw);
                return;
            }
            // the same title as when the task itself fails (RunTask); no file names on the surface (they are behind Show
            // details). What to do follows the cause: a running game holding the data, a full disk, Windows refusing the
            // folder, else the file(s) themselves
            const wstring what = Has(why, L"could not be moved") ? L"A game that uses Ray Reconstruction may still be running. Close it and try again."
                                 : Has(why, L"disk is full") ? L"The drive with your user folder is full. Free some space and try again."
                                 : Has(why, L"denied") ? L"Windows didn't allow WrongGPU to save the data in your user folder. Try again; if it keeps happening, collect the logs and report it."
                                 : Has(why, L"could not be saved") ? L"WrongGPU couldn't save the data in your user folder. Check that the disk has free space, then try again."
                                 : chose ? L"That file can't be used. Choose the DLSS Ray Reconstruction file of a recent game that has Ray Reconstruction."
                                         : L"No DLSS Ray Reconstruction file WrongGPU can use was found. Install or update a recent game that has Ray Reconstruction, or choose its file in Settings.";
            InformDetails(L"The Ray Reconstruction data couldn't be converted.\n\n" + what + (had ? L" The data in use stays." : L""), kIconError, why);
        };
    });
}

// "Convert again" / "Choose a DLSS file..." (a worker thread): a new model is made while the one in use stays. It is only set
// aside for the search (so the search looks for a DLSS file instead of reusing it), and put back unless the new one was
// written and checked: a search that finds nothing (the game uninstalled, a library offline) leaves the player's model.
// On success `msg` is left empty and `again` says a model was replaced: the window words it once it has read the new
// model's source (ModelConvertedWords, the Library's names); the core's case (the file's path, the output filter) is logged.
// The model data a search set aside (model.wgcm.keep) and never put back (the process ended half way: Task Manager, a power
// cut, a forced shutdown): back in its place at start-up, before anything reads the model (only where no model is; a model
// in its place that fails its check is dealt with by the next Convert again, which can hash it off the window's thread).
void RecoverModelKeep() {
    const wstring file = ModelFile(g.ctx), kept = file + L".keep";
    if (IsFile(file) || !IsFile(kept)) return;
    if (MoveFileExW(kept.c_str(), file.c_str(), MOVEFILE_WRITE_THROUGH)) LogLine(g.ctx, L"Model M: put back from " + kept + L" (an earlier conversion did not finish)");
}
void ConvertModelAgain(const wstring& dll, bool noScan, wstring& msg, bool& ok, bool& again) {
    const std::vector<StoreGame> libs = noScan ? std::vector<StoreGame>{} : ScanLibraries();
    const wstring file = ModelFile(g.ctx), kept = file + L".keep";
    if (IsFile(kept)) {  // left by a conversion that never ended: the model it set aside comes first, unless the one in its place is valid
        wstring why;
        if (!IsFile(file) || !ExistingModelValid(g.ctx, &why)) {
            if (MoveFileExW(kept.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) LogLine(g.ctx, L"Model M: put back from " + kept + L" (an earlier conversion did not finish)");
        } else DeleteFileW(kept.c_str());
    }
    const bool had = IsFile(file);
    if (had && !MoveFileExW(file.c_str(), kept.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        msg = L"Model M: the current model data could not be moved (" + ErrText(GetLastError()) + L"). Nothing changed.";
        return;
    }
    auto putBack = [&] { if (had) MoveFileExW(kept.c_str(), file.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH); };
    try {
        const ModelState ms = FindModel(g.ctx, nullptr, dll, libs);
        if (!ms.ok()) { putBack(); msg = L"Model M: " + ms.error + (had ? L" The model data in use stays." : L""); return; }
        Ui ui = QuietUi();
        LogLine(g.ctx, L"Model M: " + WriteModel(g.ctx, ms, ui).outcome);
        ok = true; again = had;
        if (had) DeleteFileW(kept.c_str());
    } catch (const SetupError& x) {
        putBack();
        msg = x.msg + (had ? L" The model data in use stays." : L"");
    } catch (...) {  // anything else (out of memory while converting): the model in use back first, then RunTask's note
        putBack();
        throw;
    }
}

// The settings the installed copy starts with, saved again on the window's thread once its install ended: a change made
// while it worked (the window stays live meanwhile) goes with it too
void SaveSettingsForInstalledCopy() {
    try { SaveGlobal(ResolvePaths(Join(DefaultInstallDir(), L"WrongGPU.exe"), gl.dataOverride), gl.global); }
    catch (const SetupError& e) { LogLine(g.ctx, L"self-install: the settings were not copied to the installed launcher: " + e.msg); }
}
// The 0.5 installs the Library lists that the launcher does not manage yet (Install for this user adopts them too)
std::vector<wstring> AdoptableDirs() {
    std::vector<wstring> d;
    for (auto& e : v.games) if (e.hasRecord && e.adoptable) d.push_back(e.dir);
    return d;
}

// A newer release (the start-up check, Check now): what it is and a way to its page, opened in the browser (nothing is
// downloaded from here).
void ShowUpdateNote(const wstring& message, const wstring& page) {
    const wstring text = message + L"\n\nYou have WrongGPU " + g.ctx.data->version + L". " +
                         (page.empty() ? L"Download it from the same place you got WrongGPU." : L"The release page has the download and what changed.");
    std::vector<DialogButton> b;
    if (page.empty()) b.push_back({L"Close", 0, true});
    else { b.push_back({L"Not now", 0}); b.push_back({L"Open release page", 1, true}); }
    if (RunDialog(0, text, b, 0) == 1) ShellExecuteW(v.hwnd, L"open", page.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

// The opt-in release check at start-up (and once the first run ends): a note when there is a newer release.
void StartUpdateCheck() {
    if (!gl.global.updateCheck || gl.global.updateUrl.empty() || !v.live) return;
    HWND hwnd = v.hwnd;
    const GlobalSettings gs = gl.global;
    Worker([hwnd, gs] {
        auto* up = new UpdateInfo;
        try { *up = CheckForUpdate(gs, g.ctx.data->version); }
        catch (...) { *up = UpdateInfo(); up->message = L"The release check didn't finish. Try again later."; }
        if (!PostMessageW(hwnd, WM_APP_UPDATE, 0, (LPARAM)up)) delete up;
    });
}


// ---- settings page actions -------------------------------------------------------------------------------------------
void SaveUrl(bool check) {
    wstring why;
    if (!SetGlobal(gl.global, "update_url", ToUtf8(v.urlText), &why)) { ShowToast(why, 2); return; }
    const bool saved = SaveSettings();
    if (!check) { if (saved) ShowToast(L"Release address saved.", 1); return; }
    if (v.checkingUpdate) return;
    // the request on a worker thread (the window keeps painting); its answer is a toast (WM_APP_UPDATE, wParam 1)
    v.checkingUpdate = true;
    Invalidate();
    HWND hwnd = v.hwnd;
    const GlobalSettings gs = gl.global;
    Worker([hwnd, gs] {
        auto* up = new UpdateInfo;
        try { *up = CheckForUpdate(gs, g.ctx.data->version); }  // a throw: "Checking..." still ends, in words
        catch (...) { *up = UpdateInfo(); up->message = L"The release check didn't finish. Try again later."; }
        if (!PostMessageW(hwnd, WM_APP_UPDATE, 1, (LPARAM)up)) delete up;
    });
}

// The release address shown in Settings' field and used by Save and Check now: the same text in both
void SetUrlText(const wstring& t) {
    v.urlText = t;
    if (!v.edUrl.h) return;
    wchar_t b[2048] = L"";
    GetWindowTextW(v.edUrl.h, b, 2048);
    if (t != b) SetWindowTextW(v.edUrl.h, t.c_str());
}

// Leaving a game's page with settings that differ from its profile: the player decides (Apply, Discard or stay). True:
// the page may be left.
bool LeaveGamePage() {
    if (v.page != P_GAME || !Dirty() || !v.live) return true;
    const bool running = v.cur.status == GameStatus::Running;
    const bool wait = !running && PageWritesWait();  // as the capsule's Apply
    const wstring text = L"Apply your changes to " + v.cur.name + L"?\n\nThe settings you changed on this page aren't saved yet. " +
                         (wait ? L"Apply waits until " + BusyDoneWords() + L": stay on the page, or discard them."
                          : running ? L"Apply waits until the game closes: stay on the page, or discard them." : L"Discard goes back to the settings applied last.");
    std::vector<DialogButton> b{{L"Cancel", 0}, {L"Discard", 2}};
    if (!wait && !running) b.push_back({L"Apply", 1, true, false, true});  // Apply: red, as on the capsule (it needs you)
    const int a = RunDialog(0, text, b, 0);
    if (a == 0) return false;
    if (a == 1) {
        OnApply();
        return !Dirty();  // a value Apply refused (its note was shown): the page stays
    }
    v.edit = v.saved;
    v.optiValues.clear();
    for (auto& r : v.opti.editable) v.optiValues.push_back(r.value);
    return true;
}

// false: the player kept the game's page (its unsaved changes)
bool Navigate(Page p) {
    v.menu.open = false;
    if (p != P_GAME && !LeaveGamePage()) return false;
    if (p != P_GAME) { v.pageReading = false; v.rrcfgReveal.clear(); }  // its answer is dropped (WM_APP_PAGE); a "Show me how" left behind
    if (p != v.page) { v.pageChangedAt = NowMs(); v.wipeAt.clear(); v.wipeOut.clear(); }
    // back to the Library from a game's page: the focus on the lockup it came from (not lost at the top of the page), unless
    // it is on the toolbar's Library tab; brought into view after the frame (revealFocus)
    if (p == P_LIBRARY && v.page == P_GAME && v.curIndex >= 0 && v.curIndex < kTileSpan && !ToolbarHit(v.focus)) { v.focus = ID_TILE + v.curIndex; revealFocus = true; }
    v.page = p;
    v.editFocusId = 0;
    if (p == P_GAME) return true;
    Invalidate();
    return true;
}

// ---- words for an operation's result (SHIP-BAR 4): the game by its name, the action by the button's word, a full stop; the
// core's own text (folders, file names, Windows' error text) goes behind "Show details" and into the log; the plain words
// for a success and a failure and a warning (OpSuccess, Sentence, Has, OpFailLead, OpFailTitle, PlainCause, PlainWarning) are in
// ui_words.cpp, with the step words (StepWords) and Play's unsaved question --------------------------------------------------
// The operation running now was started by a control whose result has words of its own (the RR row's Repair, the
// outside-change note's two buttons): which, and for which game folder; taken (and cleared) as it ends
int resultAs = RES_PLAIN;
wstring resultAsDir;
void SayResultAs(int kind, const wstring& dir) { resultAs = kind; resultAsDir = dir; }

// A note whose technical text waits behind "Show details" (shown in a note of its own, where a long text scrolls)
void InformDetails(const wstring& text, int icon, const wstring& details) {
    if (details.empty()) { Inform(text, icon); return; }
    if (RunDialog(icon, text, {{L"Show details", 2}, {L"Close", 1, true}}, 1) == 2) Inform(L"Details\n\n" + details, 2);
}
// The same for a note that arrives on its own (InformLater)
void NoteLater(const wstring& text, int icon, const wstring& details) {
    if (!v.live || !v.hwnd) return;
    pendingNotes.push_back({text, icon, details});
    if (modalDepth == 0) PostMessageW(v.hwnd, WM_APP_NOTE, 0, 0);
}

void LogsSaved(const Bundle& b) {
    PWSTR desk = nullptr;
    bool desktop = false;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Desktop, 0, nullptr, &desk)) && desk) desktop = IEq(Parent(b.zip), desk);
    if (desk) CoTaskMemFree(desk);
    wstring text = LogsSavedText(b.zip, desktop);
    for (auto& n : b.notes) text += L"\n\n" + Sentence(n);
    if (RunDialog(2, text, {{L"Done", 0}, {L"Show in File Explorer", 1, true}}, 0) == 1) {
        const wstring args = L"/select,\"" + b.zip + L"\"";
        ShellExecuteW(nullptr, L"open", L"explorer.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
    }
}

}  // namespace win
using namespace win;  // the window's own helpers (a named namespace: ui_render.cpp, the offscreen frames, uses some)

// ---- lui entry points the widgets call ------------------------------------------------------------------------------------
namespace lui {
void Invalidate() {
    if (!v.hwnd || !v.live) return;
    // a frame; while an entrance or a fade moves, each frame asks for the next once it is presented (WM_PAINT)
    InvalidateRect(v.hwnd, nullptr, FALSE);
}
bool Confirm(const wstring& text, int icon, const wstring& yes, const wstring& no, bool danger) {
    return RunDialog(icon, text, {{no, 0, false, false}, {yes, 1, true, danger || icon == 3}}, 0) == 1;
}
void Inform(const wstring& text, int icon) { RunDialog(icon, text, {{L"Close", 1, true, false}}, 1); }  // a note's one button: the default (accent)
void InformLater(const wstring& text, int icon) { NoteLater(text, icon, L""); }  // after the message being handled (else when the modal loop ends)
void OffThread(std::function<void()> f) { Worker(std::move(f)); }
void NoteFrom(HWND w, const wstring& t, int icon) { auto* n = new Note{t, icon, L""}; if (!PostMessageW(w, WM_APP_NOTE, 0, (LPARAM)n)) delete n; }
const std::vector<Hit>& CurrentHits() { return v.hits; }
wstring BuildLine() {
    return L"WrongGPU " + wstring(g.ctx.data->version) + L" \u00b7 " + (gl.paths.portable || v.fixture ? L"portable" : LauncherInstalled() ? L"installed" : L"not installed");
}
void ShowOpened(const GameEntry& e, const PageData* page) {
    const bool wasReading = v.pageReading && v.page == P_GAME && IEq(v.cur.dir, e.dir);
    ShowEntryWith(e, false, page);
    lastSettingId = 0;  // nothing changed on this page yet
    if (wasReading && v.focus) return;  // shown already: the keys stay where the player took them
    v.focus = ID_G_BACK;  // the page's first control has the keyboard (the ring only after keys), not nothing; UiaFrame says the game's name
    v.editFocusId = 0;  // a field still waiting for its keys (the search scrolled out with them) is not this page's: Backspace goes back
}
// A game's page opens once a worker has read its files (the game's disk may be asleep: the window keeps answering and
// painting meanwhile, the pointer shows it is working). The latest ask wins; it opens from the page it was asked on only.
void OpenGame(int index) {
    if (index < 0 || index >= (int)v.games.size()) return;
    const GameEntry e = v.games[index];
    if (!v.live || !v.hwnd || v.fixture) { ShowOpened(e, nullptr); return; }  // the ui-dump and the offscreen frames: at once
    const unsigned seq = ++pageSeq;
    pageWaiting = seq;
    pageAskedOn = v.page;
    pageAsked = e;
    pageAskedIndex = index;
    SetTimer(v.hwnd, 10, kPageReadMs, nullptr);
    HWND hwnd = v.hwnd;
    const Profile defaults = gl.global.defaults;
    const unsigned changes = DirChanges(e.dir);
    Worker([hwnd, seq, e, defaults, changes] {
        auto* m = new PageMsg;
        m->seq = seq; m->e = e; m->changes = changes;
        ReadPageDataSafe(e, defaults, m->page);
        if (!PostMessageW(hwnd, WM_APP_PAGE, 0, (LPARAM)m)) delete m;
    });
    // the pointer says it is working while the disk answers
    POINT pt; GetCursorPos(&pt);
    if (WindowFromPoint(pt) == v.hwnd) SetCursor(LoadCursorW(nullptr, IDC_APPSTARTING));
}

void MenuCommand(int owner, int cmd) {
    if (owner == ID_S_MODELMORE) { if (cmd == ID_S_MODELFILE || cmd == ID_S_RECONVERT) Activate(cmd); return; }  // Settings > Model data > Options...
    if (owner == ID_S_RRMORE) { if (cmd == ID_S_RRFILE || cmd == ID_S_RRCONVERT) Activate(cmd); return; }  // ... > Ray Reconstruction data > Options...
    const GameEntry e = v.cur;
    if (owner == ID_G_FILTER) {  // output filter
        static const char* const f[] = {"default", "original", "analytic", "quality5", "quality9"};
        if (cmd >= M_FILTER0 && cmd < M_FILTER0 + 5) SetSetting(v.edit, "output_filter", f[cmd - M_FILTER0], nullptr);
        Invalidate();
        return;
    }
    if (owner >= ID_G_OPTI0 && owner < ID_G_OPTI_END) {  // an OptiScaler key (menu key)
        const size_t row = (size_t)(owner - ID_G_OPTI0) / 10;
        if (row < v.optiValues.size() && cmd >= M_OPTI0) {
            const int k = cmd - M_OPTI0;
            if (k == 0) v.optiValues[row] = "auto";
            else { char b[16]; snprintf(b, sizeof b, "0x%02X", kMenuKeys[k - 1]); v.optiValues[row] = b; }
        }
        Invalidate();
        return;
    }
    // what opens another program's window (File Explorer, the log's editor) opens once: a double click's second click on its
    // row does not open a second one (SHIP-BAR 1)
    if (cmd == M_OPEN_FOLDER || cmd == M_OPEN_LOG) clickGuardAt = GetTickCount64();
    switch (cmd) {
        case M_OPEN_FOLDER: OpenGameFolder(e); return;  // a folder that is gone: said in words (ui_game.cpp)
        case M_OPEN_LOG: OpenGameLog(e); return;
        case M_COLLECT: {
            // off the window's thread (it reads the game, the graphics card and zips the logs: seconds), "Collecting logs..."
            // meanwhile; no operation starts until it is done, and it waits for one (a game's files are being written)
            if (v.busy) { ShowToast(L"Collect logs waits until " + BusyDoneWords() + L".", 0); return; }
            const wstring dir = e.dir;
            ShowToast(L"Collecting logs…", 0);
            RunTask(T_COLLECT, [dir]() -> std::function<void()> {
                try {
                    SystemInfo sys; sys.win = WindowsInfo(); sys.gpu = DetectGpu();
                    const Bundle b = CollectLogs(g.ctx, dir, DefaultBundlePath(dir), sys);
                    return [b] { v.toast = Toast(); LogsSaved(b); };
                } catch (const SetupError& x) {
                    const wstring why = x.msg;
                    return [why] { v.toast = Toast(); InformDetails(L"The logs couldn't be collected.\n\n" + PlainTaskCause(why, kCauseLogs), kIconError, why); };
                }
            });
            return;
        }
        case M_REPAIR: OnInstall(e, Op::Repair); return;
        case M_ROLLBACK: OnInstall(e, Op::Rollback); return;
        case M_UNINSTALL: OnInstall(e, Op::Uninstall); return;
        case M_REMOVEGAME: {  // a game added by hand: out of the Library and the next scans; nothing in its folder changes
            if (e.hasRecord || (v.busy && IEq(v.busyDir, e.dir))) return;
            if (!Confirm(L"Remove " + e.name + L" from the Library?\n\nNothing in the game's folder changes. Add game brings it back.", 0, L"Remove", L"Cancel")) return;
            auto& added = gl.global.addedGames;
            added.erase(std::remove_if(added.begin(), added.end(), [&](const wstring& x) { return IEq(FullPath(x), FullPath(e.exe)); }), added.end());
            SaveSettings();
            int gone = -1;
            for (size_t i = 0; i < v.games.size(); ++i)
                if (IEq(v.games[i].exe, e.exe)) { v.games.erase(v.games.begin() + (long long)i); gone = (int)i; break; }
            // the keyboard focus on the lockup after it in the Library's reading order (the one before it when it was the
            // last), brought into view, as every other way back to the Library has it; none there: the Library's first stop.
            // The lockups' ids are their places, which moved up by one after it.
            int next = -1;
            const auto at = std::find(libraryOrder.begin(), libraryOrder.end(), gone);
            if (gone >= 0 && at != libraryOrder.end()) {
                if (at + 1 != libraryOrder.end()) next = *(at + 1);
                else if (at != libraryOrder.begin()) next = *(at - 1);
                if (next > gone) --next;
            }
            v.focus = next >= 0 ? ID_TILE + next : 0;
            if (next >= 0) revealFocus = true;
            else focusFirstStop = true;
            v.editFocusId = 0;
            ForgetPlaces();
            v.page = P_LIBRARY; v.pageChangedAt = GetTickCount64(); v.curIndex = -1;
            ShowToast(e.name + L" was removed from the Library.", 1);
            Invalidate();
            return;
        }
        case M_RESET: {
            const Profile d = DefaultProfile();
            // the Upscaler section's settings only: the FPS overlay is OptiScaler's (its section, and "Reset OptiScaler settings...")
            for (auto* k : {"wronggpu", "executor", "output_filter", "pre_exposure", "sparkle_guard"}) SetSetting(v.edit, k, d.get(k), nullptr);
            ShowToast(v.edit.v == v.saved.v ? L"These settings are the defaults already." : L"Default settings are shown. Apply to use them.", 0);
            Invalidate();
            return;
        }
        case M_OPTI0 - 1: {  // reset the OptiScaler settings changed here
            if (PageWritesWait() || e.status == GameStatus::Orphaned) return;  // the menu item is off then
            if (!Confirm(L"Reset the OptiScaler settings you changed in " + e.name + L"?\n\nThey go back to the values WrongGPU's install wrote.", 0, L"Reset", L"Cancel", true)) return;
            if (v.busy || v.listCached) return;  // something started while the question was open
            OpRequest r; r.op = Op::Apply; r.dir = e.dir; r.optiReset = r.optiOnly = true;
            StartOp(r);
            return;
        }
        default: return;
    }
}

// A text field is wanted now (Tab, a click, Enter) while the page makes its entrance: the entrance ends, so the native
// field shows (SyncEdits) and the first keys go into it.
void FieldNow() {
    if (!PageEntering()) return;
    v.pageChangedAt = 0;
    Invalidate();
    if (v.hwnd) UpdateWindow(v.hwnd);
}

// The first run ends: the Library, and the opt-in release check (start-up's own check waits for the first run)
void FinishFirstRun() {
    gl.global.firstRunDone = true;
    SettleFirstRunLog();  // the choice is made: the first run's log lines join launcher.log in the data folder it gave
    SaveSettings();  // not saved: the Library opens anyway (the note says why; the next start asks again)
    v.first.open = false;
    v.page = P_LIBRARY;
    // the keyboard is not left on the sheet's Continue, which is gone: the Library's first stop has it once drawn
    v.focus = 0;
    focusFirstStop = true;
    clickGuardAt = GetTickCount64();  // a double click on Continue does not open the lockup under it
    StartUpdateCheck();
    Invalidate();
}

// A click, Enter or Space on a widget (ActivateWidget). On a game's page the control that changed a setting is remembered
// (a select's menu item counts as its select): the focus goes back to it when the unsaved capsule closes (PaintAll).
void Activate(int id) {
    // Close pressed during an operation (the window was to close once it ends: WM_CLOSE) is taken back when the player goes on
    // using the window: any control but a dialog's buttons and the window's own (a second Close keeps the promise)
    if ((v.closeAfterOp || closeAfterNotes) && !(id >= ID_DIALOG && id < ID_DIALOG + 100) && !(id >= ID_CAP_MIN && id <= ID_CAP_CLOSE)) {
        v.closeAfterOp = closeAfterNotes = false;
        ShowToast(L"WrongGPU stays open.", 0);
    }
    if (v.page != P_GAME || id == ID_G_APPLY || id == ID_G_DISCARD) { ActivateWidget(id); return; }
    const int by = id >= ID_MENU && id < ID_MENU + 100 ? v.menu.owner : id;
    const std::map<string, string> edit = v.edit.v;
    const std::vector<string> opti = v.optiValues;
    ActivateWidget(id);
    if (v.page == P_GAME && (v.edit.v != edit || v.optiValues != opti)) lastSettingId = by;
}

// ---- painting --------------------------------------------------------------------------------------------------------------
void PaintAll(Gfx& gx) {
    v.hits.clear();
    v.barMoving = false;  // set again by a progress bar that moves as this frame draws it
    v.tips.clear();
    v.wipeArts.clear();  // the art that can wipe, and the one the pointer wipes: set again as the page draws them
    v.wipeHot = 0;
    for (auto* e : {&v.edSearch, &v.edUrl}) e->want = false;
    for (auto& e : v.edOpti) e.want = false;
    gx.Clear(pal.bg);
    cv = ContentView{Rc{0, 0, v.W, v.H}, 0};
    const Page p = v.page;
    // the page's area: everything under the toolbar (option E has no rail); the toolbar is drawn once the scroll is known
    const Rc area{0, kBarH, v.W, v.H - kBarH};
    const float maxScroll = std::max(0.f, v.contentH[p] - v.H);
    v.target[p] = std::min(std::max(0.f, v.target[p]), maxScroll);
    v.scroll[p] = std::min(std::max(0.f, v.scroll[p]), maxScroll);
    DrawToolbar(gx);
    cv = ContentView{area, std::round(v.scroll[p] * gx.scale) / gx.scale};
    v.pageCx = 0;  // the page sets it (toasts centre on it)
    // a page's entrance: its content fades in as it rises 8 DIPs (the toolbar stays put); none with Animation effects off
    const float pe = Entrance(v.pageChangedAt, kPageMs);
    gx.PushClip(area);
    if (pe < 1) gx.PushFade(pe);
    gx.SetOffset(0, -cv.scroll + std::round(8 * (1 - pe) * gx.scale) / gx.scale);
    if (p == P_LIBRARY) PaintLibrary(gx, area);
    else if (p == P_GAME) PaintGame(gx, area);
    else PaintSettings(gx, area);
    gx.SetOffset(0, 0);
    if (pe < 1) gx.PopFade();
    gx.PopClip();
    FocusRingOverBar(gx);  // a control at the page's top: its ring over the toolbar's edge too, not cut flat
    DrawScrollbar(gx, p);
    cv = ContentView{Rc{0, 0, v.W, v.H}, 0};
    // the upscale wipe's fades: only for art this frame drew (scrolled away, another page, a scan: the next hover fades in)
    for (auto it = v.wipeAt.begin(); it != v.wipeAt.end();) {
        bool drawn = false;
        for (auto& a : v.wipeArts) drawn |= a.id == it->first;
        it = drawn ? std::next(it) : v.wipeAt.erase(it);
    }
    for (auto it = v.wipeOut.begin(); it != v.wipeOut.end();) {
        bool drawn = false;
        for (auto& a : v.wipeArts) drawn |= a.id == it->first;
        it = drawn ? std::next(it) : v.wipeOut.erase(it);
    }
    if (p == P_GAME) PaintGameOverlay(gx, area);
    if (v.first.open) {
        v.hits.clear();
        PaintFirstRun(gx);
        // the sheet for UI Automation: a dialog named by its title that holds everything on it (ui_uia.cpp), before its
        // contents (they take the clicks). Its place: PaintFirstRun's width, from its top to under its foot.
        const float sw = std::min(560.f, v.W - 32);
        Hit sheet;
        sheet.id = ID_FR_SHEET; sheet.role = R_DIALOG; sheet.name = L"Welcome to WrongGPU"; sheet.focusable = false;
        float bottom = v.first.bodyTop;
        for (auto& h : v.hits) bottom = std::max(bottom, h.r.b());
        sheet.r = sheet.full = Rc{std::round((v.W - sw) / 2), v.first.bodyTop - 1, sw, std::min(v.H, bottom + 22) - (v.first.bodyTop - 1)};
        v.hits.insert(v.hits.begin(), sheet);
        // the focus on a button the sheet took away once its checks were done (Choose a DLSS file... and Search your
        // games when the new check found a model): Continue, the sheet's default, has it (UiaFrame tells UI Automation),
        // not nothing.
        // While the checks run nothing moves it (the sheet has the focus then; the buttons may come back).
        if (!v.first.busy && !v.dialog.open && v.focus) {
            bool there = false, go = false;
            for (auto& h : v.hits) { there |= h.id == v.focus; go |= h.id == ID_FR_CONTINUE && h.enabled; }
            if (!there && go) v.focus = ID_FR_CONTINUE;
        }
    }
    DrawToast(gx);  // over the first-run sheet too (it takes no clicks: the sheet's hits stay)
    // an open menu over the toast (the top layer, as a native menu is): a toast never covers its items (the minimum window,
    // where the toast rises into a long menu's last rows); it fades under the menu as it does under the pointer (DrawToast)
    if (!v.first.open) DrawMenu(gx);
    if (v.dialog.open) { v.hits.clear(); DrawDialog(gx); }
    DrawCaption(gx);  // the window's own buttons, over everything (a sheet's scrim too)
    // reading order within a row: right-aligned rows (toolbars, a bay's actions, the plate, dialog buttons) are drawn right
    // to left, so each run of consecutive hits on one line is put left to right. Tab, UI Automation and the dump follow
    // v.hits; the order between rows stays the drawing order. Two hits that overlap are never swapped (a bay and its
    // buttons, the search field and its clear button, a menu over the page), so hit testing (last hit wins) is unchanged.
    for (size_t a = 0; a < v.hits.size();) {
        size_t b = a + 1;
        while (b < v.hits.size() && std::fabs(v.hits[b].r.cy() - v.hits[a].r.cy()) < 6) ++b;
        for (size_t i = a + 1; i < b; ++i)
            for (size_t j = i; j > a; --j) {
                const Rc &left = v.hits[j - 1].r, &right = v.hits[j].r;
                if (left.x <= right.x || (Intersect(left, right).w > 0 && Intersect(left, right).h > 0)) break;
                std::swap(v.hits[j - 1], v.hits[j]);
            }
        a = b;
    }
    // the keyboard focus on a control that is gone (a pill that became a progress bar, Install that became Play, a game that
    // left the Library, "Update N games" turned off): the stop nearest to where it was on the same page, so the ring and
    // the next Tab go on from there. Another page: no focus (the next Tab starts at the top). Dialogs and the first run
    // keep their own focus.
    static Rc focusRc;
    static int focusPage = -1;
    if (p == P_LIBRARY && !v.first.open && !v.dialog.open) {
        libraryOrder.clear();
        for (auto& h : v.hits) if (h.id >= ID_TILE && h.id < ID_TILE + kTileSpan) libraryOrder.push_back(h.id - ID_TILE);
        // "Show N more" from the keyboard: the first lockup it revealed (above the button now) takes the focus, into view
        if (attnReveal) {
            attnReveal = false;
            for (auto& h : v.hits) {
                if (h.id == ID_ATTNMORE) break;
                if (h.id >= ID_TILE && h.id < ID_TILE + kTileSpan && h.focusable && h.enabled &&
                    std::find(attnShownBefore.begin(), attnShownBefore.end(), h.id) == attnShownBefore.end()) {
                    const Hit t = h;
                    FocusHit(&t);
                    break;
                }
            }
            if (v.focus == ID_ATTNMORE) revealFocus = true;  // none found: the button at least stays in view
            attnShownBefore.clear();
        }
    }
    // the first run ended (FinishFirstRun): the Library's first stop (the search field, else the first lockup or an empty
    // state's button) takes the focus, not nothing, and UI Automation hears it
    if (focusFirstStop && !v.first.open && !v.dialog.open && p == P_LIBRARY) {
        focusFirstStop = false;
        for (auto& h : v.hits) {
            if (!h.focusable || !h.enabled || ToolbarHit(h.id) || h.id == ID_SCROLLBAR) continue;
            v.focus = h.id;
            if (h.role == R_EDIT) v.editFocusId = h.id;  // its native EDIT takes the keys once shown (SyncEdits; it says so itself)
            else UiaFocusChanged(h.id);
            break;
        }
    }
    // the Library's place the keyboard was on when last drawn, and its game (a place's id is its game's place in the list, the
    // banner's ids name whichever game it shows)
    static int trackId = 0;
    static wstring trackGame;
    if (v.focus && !v.dialog.open && !v.first.open) {
        // the same id names another game now (the banner took another one): the focus goes with its game (ui_uia.cpp)
        if (p == P_LIBRARY && focusPage == v.page) FollowPlaceGame(trackId, trackGame);
        const Hit *at = nullptr, *still = nullptr;
        for (auto& h : v.hits) {
            if (h.id != v.focus) continue;
            if (h.focusable && h.enabled) at = &h;
            else if (!h.enabled && h.role != R_TEXT) still = &h;
        }
        // a control that went busy or off where it is ("Update 1 game" -> "Updating...", Convert -> "Converting...", Collect
        // logs while an operation runs): the focus stays on it, shown off (Return and Space do nothing there, UI Automation
        // hears its new name), so a second press never lands on whatever was nearest (SHIP-BAR: Enter-repeat never starts
        // another action), and Tab goes on from its place (MoveFocus)
        if (!at && still && focusPage == v.page) at = still;
        // back on the Library from the page of the featured banner's game: it has no lockup there, so its banner's Game
        // settings (the button that opened the page) takes the focus, not nothing. A lockup's action whose operation made
        // its game the banner's (Update on the game played last): the banner's action (Play), not the next game's pill that
        // slid into its place
        if (!at && p == P_LIBRARY && v.featIndex >= 0 && (v.focus == ID_TILE + v.featIndex || v.focus == ID_TILEQA + v.featIndex)) {
            const int want = v.focus == ID_TILEQA + v.featIndex ? ID_FEAT0 + 1 : ID_FEAT0 + 2;
            const Hit* to = nullptr;
            for (auto& h : v.hits) if (!to && h.id == want && h.focusable && h.enabled) to = &h;
            for (auto& h : v.hits) if (!to && h.id == ID_FEAT0 + 2 && h.focusable && h.enabled) to = &h;  // its action is off (Play blocked)
            if (to) { v.focus = to->id; at = to; UiaFocusChanged(to->id); revealFocus = true; }
        }
        // the unsaved capsule closed under the focus (Apply, Discard): the control a setting was last changed with, not the
        // stop nearest to where the capsule floated
        if (!at && focusPage == v.page && p == P_GAME && (v.focus == ID_G_APPLY || v.focus == ID_G_DISCARD) && lastSettingId)
            for (auto& h : v.hits)
                if (h.id == lastSettingId && h.focusable && h.enabled) {
                    v.focus = h.id; at = &h; revealFocus = true;
                    if (h.role == R_EDIT) v.editFocusId = h.id;  // its native EDIT takes the keys once shown (SyncEdits; it says so itself)
                    else UiaFocusChanged(h.id);
                    break;
                }
        RevealIfMoved(at, focusPage == v.page);  // it moved on the page itself (another shelf): into view (ui_uia.cpp)
        if (at) { focusRc = at->r; focusPage = v.page; }
        else if (focusPage == v.page) {
            const Hit* best = nullptr;
            float bd = 1e18f;
            for (auto& h : v.hits) {
                if (!h.focusable || !h.enabled || ToolbarHit(h.id) || (h.id >= ID_MENU && h.id < ID_MENU + 100)) continue;
                const float dx = h.r.cx() - focusRc.cx(), dy = h.r.cy() - focusRc.cy();
                const float d = dx * dx + dy * dy + (h.shown ? 0.f : 1e12f);  // one in view first
                if (d < bd) { bd = d; best = &h; }
            }
            if (best) { v.focus = best->id; focusRc = best->r; UiaFocusChanged(best->id); }
            else v.focus = 0;
        } else v.focus = 0;
    }
    if (!v.dialog.open && !v.first.open) {  // under a dialog it stays as it was when the dialog opened
        trackId = p == P_LIBRARY ? v.focus : 0;
        trackGame = trackId ? PlaceGame(trackId) : wstring();
    }
}

}  // namespace lui

namespace win {

// ---- the native text fields ------------------------------------------------------------------------------------------------
std::map<HWND, wstring> editNames;  // the accessible name each EDIT was given (UiaNameEdit)
HWND editFocusBack = nullptr;       // the EDIT that had the keys when the window was deactivated (they go back to it)
LRESULT CALLBACK EditProc(HWND h, UINT m, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR) {
    if (m == WM_GETDLGCODE) return DLGC_WANTCHARS | DLGC_HASSETSEL | DLGC_WANTARROWS;
    if (m == WM_NCDESTROY) { UiaForgetEdit(h); editNames.erase(h); if (editFocusBack == h) editFocusBack = nullptr; }
    // an open menu (opened from UI Automation while the keys were here) takes the keys; a click on the field only closes it
    if (v.menu.open && (m == WM_KEYDOWN || m == WM_SYSKEYDOWN || m == WM_CHAR || m == WM_LBUTTONDOWN)) {
        if (m == WM_LBUTTONDOWN) { v.menu.open = false; Invalidate(); }
        else if (m != WM_CHAR) { SetFocus(GetParent(h)); SendMessageW(GetParent(h), m, wp, lp); }
        return 0;
    }
    // the window's own keys: Alt+Left (back), Ctrl+1 / Ctrl+2 (the pages)
    if (m == WM_SYSKEYDOWN && wp == VK_LEFT) { SendMessageW(GetParent(h), m, wp, lp); return 0; }
    if (m == WM_KEYDOWN && (wp == '1' || wp == '2') && GetKeyState(VK_CONTROL) < 0) { SendMessageW(GetParent(h), m, wp, lp); return 0; }
    if (m == WM_KEYDOWN && wp == 'S' && GetKeyState(VK_CONTROL) < 0) { SendMessageW(GetParent(h), m, wp, lp); return 0; }  // Apply (Ctrl+S)
    if (m == WM_KEYDOWN && (wp == VK_TAB || wp == VK_ESCAPE || (wp == 'F' && GetKeyState(VK_CONTROL) < 0) || wp == VK_RETURN)) {
        if (wp == VK_ESCAPE && h == v.edSearch.h && GetWindowTextLengthW(h) > 0) { SetWindowTextW(h, L""); return 0; }
        if (wp == VK_ESCAPE) return 0;  // Escape belongs to the field (Windows and macOS): it never leaves the page from in it
        if (wp == VK_RETURN && h == v.edUrl.h) { if (!(lp & (1 << 30))) Activate(ID_S_SAVEURL); return 0; }  // once per press (OnKey's rule): a held key saves once
        if (wp == VK_RETURN) { if (!(lp & (1 << 30))) KeyApply(); return 0; }  // an OptiScaler number: the capsule's Apply, its default
        SendMessageW(GetParent(h), WM_KEYDOWN, wp, lp);
        return 0;
    }
    if (m == WM_CHAR && (wp == VK_TAB || wp == VK_RETURN || wp == VK_ESCAPE)) return 0;
    if (m == WM_SETFOCUS || m == WM_KILLFOCUS) Invalidate();
    return EditCueProc(h, m, wp, lp);  // its placeholder in the theme's text2 (ui_frame.cpp), then the EDIT's own
}

void MakeFonts() {
    // the new fonts go to the fields before the old ones are deleted (a field never holds a deleted handle)
    const HFONT oldEdit = editFont;
    auto font = [](float size, const wchar_t* face, const wchar_t* fallback) {
        LOGFONTW lf{}; lf.lfHeight = -Px(size); lf.lfWeight = FW_NORMAL; lf.lfQuality = CLEARTYPE_QUALITY; lf.lfCharSet = DEFAULT_CHARSET;
        wcsncpy(lf.lfFaceName, face, LF_FACESIZE - 1);
        HFONT f = CreateFontIndirectW(&lf);
        wchar_t got[LF_FACESIZE] = L"";
        HDC dc = GetDC(nullptr); HGDIOBJ o = SelectObject(dc, f); GetTextFaceW(dc, LF_FACESIZE, got); SelectObject(dc, o); ReleaseDC(nullptr, dc);
        if (_wcsicmp(got, face)) { DeleteObject(f); wcsncpy(lf.lfFaceName, fallback, LF_FACESIZE - 1); f = CreateFontIndirectW(&lf); }
        return f;
    };
    editFont = font(13, L"Segoe UI Variable Text", L"Segoe UI");
    // the fields' line heights, measured here once (not on every frame: SyncEdits)
    auto lineH = [](HFONT f) { TEXTMETRICW t{}; HDC dc = GetDC(nullptr); HGDIOBJ o = SelectObject(dc, f); GetTextMetricsW(dc, &t); SelectObject(dc, o); ReleaseDC(nullptr, dc); return (int)t.tmHeight; };
    editFontH = lineH(editFont);
    // every field in the font it was made with (a DPI change keeps the UI font: the fields stay on their painted boxes)
    for (auto* e : {&v.edSearch, &v.edUrl}) if (e->h) SendMessageW(e->h, WM_SETFONT, (WPARAM)editFont, TRUE);
    for (auto& e : v.edOpti) if (e.h) SendMessageW(e.h, WM_SETFONT, (WPARAM)editFont, TRUE);
    if (oldEdit) DeleteObject(oldEdit);
}

HWND MakeEdit(int id, const wstring& text) {
    HWND h = CreateWindowExW(0, L"EDIT", text.c_str(), WS_CHILD | WS_TABSTOP | ES_AUTOHSCROLL, 0, 0, 10, 10, v.hwnd, (HMENU)(INT_PTR)id, g.hinst, nullptr);
    if (!h) return nullptr;
    SendMessageW(h, WM_SETFONT, (WPARAM)editFont, TRUE);
    SendMessageW(h, EM_SETMARGINS, EC_LEFTMARGIN | EC_RIGHTMARGIN, 0);
    SetWindowSubclass(h, EditProc, 1, 0);
    return h;
}

// After a frame: the EDIT children sit on their painted fields, hidden when their field is not on screen.
// A field the keyboard went to before it was in view (Tab to it while it scrolls in, or it was hidden under a dialog or a
// menu with the keys in it) takes the keys once it shows, if the focus is still on it (v.editFocusId).
void SyncEdits() {
    if (!v.live) return;
    auto sync = [&](EditBox& e, int id, int hitId, const wstring& text) {
        if (!e.h && e.want) e.h = MakeEdit(id, text);
        if (!e.h) return;
        // its name where the keys go ("Search games", "Release address (https)", an OptiScaler row's name): Narrator reads the
        // EDIT, not the painted field (whose name UI Automation would otherwise not have; it read the control id)
        if (!e.name.empty() && editNames[e.h] != e.name) { UiaNameEdit(e.h, e.name); editNames[e.h] = e.name; }
        const int eh = editFontH + 2;
        const Rc r = e.r;
        if (EditShown(e)) {
            const int x = Px(r.x), w = Px(r.w), y = Px(r.cy()) - eh / 2;
            RECT cur; GetWindowRect(e.h, &cur); MapWindowPoints(nullptr, v.hwnd, (POINT*)&cur, 2);
            if (cur.left != x || cur.top != y || cur.right - cur.left != w || cur.bottom - cur.top != eh) {
                SetWindowPos(e.h, nullptr, x, y, w, eh, SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOCOPYBITS);
                // painted at its new place now: after a big resize its area would otherwise show black until it repaints
                RedrawWindow(e.h, nullptr, nullptr, RDW_INVALIDATE | RDW_ERASE | RDW_FRAME | RDW_UPDATENOW);
            }
            if (!IsWindowVisible(e.h)) ShowWindow(e.h, SW_SHOWNA);
            if (v.editFocusId == hitId && v.focus == hitId && GetFocus() != e.h) { v.editFocusId = 0; SetFocus(e.h); }
        } else if (IsWindowVisible(e.h)) {
            if (GetFocus() == e.h) {
                SetFocus(v.hwnd);
                if (v.focus == hitId) v.editFocusId = hitId;  // back to it once it shows again
            }
            ShowWindow(e.h, SW_HIDE);
        }
    };
    sync(v.edSearch, kEditSearch, ID_SEARCH, v.search);
    sync(v.edUrl, kEditUrl, ID_S_URL, v.urlText);  // an address, in the UI font as the search (not code)
    for (size_t i = 0; i < v.edOpti.size(); ++i) {
        const wstring text = i < v.optiValues.size() ? FromUtf8(v.optiValues[i]) : L"";
        const bool fresh = !v.edOpti[i].h;
        sync(v.edOpti[i], kEditOpti0 + (int)i, ID_G_OPTI0 + (int)i * 10, text);  // a value, in the UI font (not code)
        UiaMarkEdit(v.edOpti[i].h, FieldInvalidWhy(ID_G_OPTI0 + (int)i * 10));  // a value Apply refused: invalid until it changes
        if (!fresh && v.edOpti[i].h && v.page == P_GAME && GetFocus() != v.edOpti[i].h) {  // another game's page: its values
            wchar_t b[256] = L""; GetWindowTextW(v.edOpti[i].h, b, 256);
            if (text != b) SetWindowTextW(v.edOpti[i].h, text.c_str());
        }
    }
}

// ---- input --------------------------------------------------------------------------------------------------------------------
const Hit* HitAt(float x, float y) {
    for (size_t i = v.hits.size(); i-- > 0;) if (v.hits[i].shown && v.hits[i].r.has(x, y)) return &v.hits[i];
    return nullptr;
}
// ui_frame.cpp: the frames while something moves (Animating, Kick: the next one asked for; Advance: one step of the smooth
// scroll and the hover lifts) and the scroll itself (ScrollBy: a glide; ScrollNow: one to one, a touchpad's steps; the last
// scroll input's time: hover follows the content once it rests)
bool Animating();
void Kick();
extern ULONGLONG lastAdvance;
void Advance();
extern ULONGLONG scrollInputAt;
constexpr ULONGLONG kHoverRestMs = 100;
void ScrollBy(float dy);
void ScrollNow(float dy);
// The hover tooltip (ui_frame.cpp): one comctl32 tooltip whose tool is the hit under the pointer
extern HWND tipWnd;
void UpdateTip();

bool GridId(int id) { return id >= ID_TILE && id < ID_TILEQA + kTileSpan; }  // a Library lockup or its action
int gridLast = 0;  // the lockup (or action) the keyboard was last on: its shelf's Tab stop
// The Tab stops in reading order (PaintAll). A group the arrows walk is one stop (Windows and macOS: Tab goes from group to
// group, the arrows inside it): a Library shelf (its lockups and their actions, between its heading and the next one or the
// next control), a radio group (a segmented control, the first run's choices: consecutive ids). The group's stop is the
// focused one, else the one the keyboard was last on (a shelf) or the chosen one (radios), else the first.
std::vector<const Hit*> Focusables() {
    std::vector<const Hit*> f;
    const size_t n = v.hits.size();
    const bool onNav = v.focus == ID_NAV_LIBRARY || v.focus == ID_NAV_SETTINGS;
    // Settings' section index: one stop too (a list: UI Automation's List of ListItems), the entry the keyboard is on, else the section in view, else
    // the first; Up and Down go between them (OnKey)
    auto indexId = [](int id) { return id >= ID_S_INDEX0 && id < ID_S_INDEX0 + 3; };
    int indexStop = 0;
    for (auto& h : v.hits)
        if (indexId(h.id) && h.focusable && h.enabled && (!indexStop || h.id == v.focus || (h.state == 1 && !indexId(v.focus)))) indexStop = h.id;
    for (size_t i = 0; i < n;) {
        const Hit& h = v.hits[i];
        // the toolbar's Library | Settings: one stop (a tab list), the selected tab or the one the keyboard is on; Left and
        // Right go between them
        if ((h.id == ID_NAV_LIBRARY || h.id == ID_NAV_SETTINGS) && (onNav ? v.focus != h.id : h.state != 1)) { ++i; continue; }
        if (indexId(h.id) && h.id != indexStop) { ++i; continue; }
        const int kind = GridId(h.id) ? 1 : h.role == R_RADIO ? 2 : 0;
        if (kind == 0 || !h.focusable || !h.enabled) {
            if (h.focusable && h.enabled) f.push_back(&h);
            ++i;
            continue;
        }
        size_t j = i + 1;
        for (; j < n; ++j) {
            const Hit& x = v.hits[j];
            if (kind == 1 ? !GridId(x.id) && (x.focusable || (x.id >= ID_SHELF0 && x.id < ID_SHELF0 + 4)) : x.role != R_RADIO || x.id != v.hits[j - 1].id + 1) break;
        }
        const Hit *pick = nullptr, *first = nullptr;
        for (size_t x = i; x < j; ++x) {
            const Hit& c = v.hits[x];
            if (!c.focusable || !c.enabled || (kind == 1 && !GridId(c.id))) continue;
            if (!first) first = &c;
            if (c.id == v.focus) pick = &c;
        }
        for (size_t x = i; x < j && !pick; ++x) {
            const Hit& c = v.hits[x];
            if (c.focusable && c.enabled && (kind == 1 ? c.id == gridLast : c.state == 1)) pick = &c;
        }
        if (pick || first) f.push_back(pick ? pick : first);
        i = j;
    }
    return f;
}
// A control brought into view: the toolbar, and on the game page its pinned plate, cover the view's top, the unsaved
// capsule's band its bottom (the toolbar's, the plate's and the capsule's own controls stay where they are). Measured at
// the scroll the page is going to (a smooth scroll still moving is counted once: holding Tab never overshoots). words:
// text too (UI Automation's ScrollIntoView: a screen reader's scan mode), not only what takes the focus.
void Reveal(const Hit* h, bool words = false) {
    if (!h || ToolbarHit(h->id) || v.dialog.open || v.first.open || v.menu.open || (Passive(h->role) && !words)) return;
    if (v.page == P_GAME && (h->id == ID_G_APPLY || h->id == ID_G_DISCARD || h->id == ID_G_UNSAVED)) return;  // the capsule floats over the page
    if (v.page == P_GAME && v.pinH > 0 && h->id >= ID_G_BACK && h->id <= ID_G_MORE) return;  // on the pinned plate
    if (v.page == P_GAME && v.pinH > 0 && h->shown && h->r.h > 0.5f && h->r.y < v.pinH - 1 && !h->sticky) return;  // the plate's own words (it stays put)
    const Page p = v.page;
    const float top = std::max(kBarH, p == P_GAME ? v.pinH : 0.f), bar = GameBarH();
    if (p == P_GAME && h->sticky && v.stickCol.on) {  // the right column: where it is at a scroll (View::stickCol), not 1:1
        const View::StickyCol& c = v.stickCol;
        auto colTop = [&](float s) { return s > c.pinAt ? std::max(c.y, std::min(s + c.stick, c.max)) : c.y; };
        const float off = h->full.y + h->scroll - colTop(h->scroll), hh = h->full.h;
        // the room above it at scroll s (the header pins past pinAt): it never grows as s grows, so a bisection finds the
        // nearest scroll that shows it
        auto above = [&](float s) { return colTop(s) + off - s - (s > c.pinAt ? std::max(kBarH, c.pinH) : kBarH); };
        const float t = v.target[p], range = std::max(0.f, v.contentH[p] - v.H), lowest = v.H - (bar > 0 ? bar + 30 : 4);
        float to = t;
        if (above(t) < 4) {  // scroll up to the largest s with 24 DIPs above it
            float lo = 0, hi = t;
            if (above(0) < 24) hi = 0;
            for (int i = 0; i < 40 && hi - lo > 0.5f; ++i) { const float m = (lo + hi) / 2; (above(m) >= 24 ? lo : hi) = m; }
            to = above(0) < 24 ? 0 : lo;
        } else if (colTop(t) + off + hh - t > lowest) {  // scroll down to the smallest s whose bottom clears the band
            const float want = v.H - (bar > 0 ? bar + 50 : 24);
            float lo = t, hi = range;
            for (int i = 0; i < 40 && hi - lo > 0.5f; ++i) { const float m = (lo + hi) / 2; (colTop(m) + off + hh - m <= want ? hi : lo) = m; }
            to = hi;
        }
        if (to != t) ScrollBy(to - t);
        return;
    }
    const float y = h->full.y + h->scroll - v.target[p], b = h->full.b() + h->scroll - v.target[p];
    if (y < top + 4) ScrollBy(y - top - 24);
    else if (b > v.H - (bar > 0 ? bar + 30 : 4)) ScrollBy(b - v.H + (bar > 0 ? bar + 50 : 24));
}
// The first run's card scrolled to show h when the window is too short for it
void FirstRunShow(const Hit* h) {
    if (v.first.open && !v.dialog.open && v.first.maxScroll > 0 && h->id != ID_FR_CONTINUE && h->id != ID_FR_SHEET) {  // Continue is on the base plate, which stays put
        if (h->r.y < v.first.bodyTop + 8) v.first.scroll += h->r.y - v.first.bodyTop - 16;
        else if (h->r.b() > v.first.bodyBottom - 8) v.first.scroll += h->r.b() - v.first.bodyBottom + 16;
        v.first.scroll = std::min(std::max(0.f, v.first.scroll), v.first.maxScroll);
    }
}
void FocusHit(const Hit* h) {
    if (!h) return;
    Hit keep;
    if (h->role == R_EDIT && PageEntering()) { keep = *h; h = &keep; FieldNow(); }  // FieldNow paints: v.hits is built again
    v.focus = h->id;
    v.focusVisible = true;
    if (GridId(h->id)) gridLast = h->id;
    if (h->role == R_EDIT) {
        HWND e = EditOf(h->id);
        if (e && IsWindowVisible(e)) { SetFocus(e); v.editFocusId = 0; }
        else {  // scrolled out or under a band: the keys go to it once it is in view (SyncEdits); the ring shows meanwhile
            v.editFocusId = h->id;
            if (GetFocus() != v.hwnd) SetFocus(v.hwnd);
        }
    } else {
        v.editFocusId = 0;
        if (GetFocus() != v.hwnd) SetFocus(v.hwnd);
    }
    // into view (content items that are cut off at the view's edge)
    FirstRunShow(h);
    Reveal(h);  // out from under the toolbar, the pinned plate and the unsaved capsule
    UiaFocusChanged(h->id);
    Invalidate();
}
}  // namespace win

// UI Automation's SetFocus (ui_uia.cpp, on the window's thread): as Tab moves it, into view and a text field's keys too;
// words (no focus to take) are brought into view
void lui::FocusId(int id) {
    for (auto& h : v.hits)
        if (h.id == id) {
            if (h.focusable && h.enabled) FocusHit(&h);
            else RevealId(id);
            return;
        }
}
// UI Automation's ScrollIntoView (ui_uia.cpp, on the window's thread): the page scrolled as Tab would scroll it, also to
// words (Narrator's scan mode reads the Last run rows, a section's sentence); the first run's card; an open menu's item
void lui::RevealId(int id) {
    if (v.menu.open && id >= ID_MENU && id < ID_MENUBOX) { MenuShowItem(id - ID_MENU); Invalidate(); return; }
    for (auto& h : v.hits)
        if (h.id == id) {
            const Hit t = h;
            FirstRunShow(&t);
            Reveal(&t, true);
            Invalidate();
            return;
        }
}

namespace win {
void MoveFocus(bool back) {
    const auto f = Focusables();
    if (f.empty()) return;
    int at = -1;
    HWND fe = GetFocus();
    for (size_t i = 0; i < f.size(); ++i) {
        if (f[i]->id == v.focus) at = (int)i;
        if (fe && fe != v.hwnd && fe == EditOf(f[i]->id)) at = (int)i;  // the keys in a text field's EDIT
    }
    const int n = (int)f.size();
    // the focus on a control that went busy or off (it is no stop now: PaintAll keeps it there): on from its place in the
    // reading order, not from the window's top
    if (at < 0 && v.focus && !v.hits.empty()) {
        const Hit* on = nullptr;
        for (auto& h : v.hits) if (h.id == v.focus) on = &h;
        if (on) {
            int j = back ? -1 : n;  // the first stop after it, or the last one before it
            for (int i = 0; i < n; ++i)
                if (back ? f[i] < on : f[i] > on) { if (back) j = i; else { j = i; break; } }
            FocusHit(f[(j + n) % n]);
            return;
        }
    }
    const int next = at < 0 ? (back ? n - 1 : 0) : (at + (back ? n - 1 : 1)) % n;
    FocusHit(f[next]);
}

// A key a menu takes (type-ahead): a letter or a digit, lower case; 0 none
wchar_t MenuChar(WPARAM k) {
    if ((k >= 'A' && k <= 'Z') || (k >= '0' && k <= '9')) return (wchar_t)towlower((wint_t)k);
    if (k >= VK_NUMPAD0 && k <= VK_NUMPAD9) return (wchar_t)(L'0' + (k - VK_NUMPAD0));
    return 0;
}
wstring menuTyped;           // the letters typed into the open menu (type-ahead), cleared after a second without a key
ULONGLONG menuTypedAt = 0;

void OnKey(WPARAM k, LPARAM lp) {
    const bool ctrl = GetKeyState(VK_CONTROL) < 0, shift = GetKeyState(VK_SHIFT) < 0, altKey = GetKeyState(VK_MENU) < 0;
    // Enter and Space act once per press: a key held down (its repeats) never answers the dialog or menu it opened, or
    // toggles a switch over and over
    const bool repeat = (lp & (1 << 30)) != 0;
    if ((k == VK_RETURN || k == VK_SPACE) && repeat) return;
    if (v.menu.open) {
        Menu& m = v.menu;
        auto step = [&](int d) {
            const int n = (int)m.items.size();
            for (int i = 1; i <= n; ++i) {
                const int j = ((m.hot < 0 ? (d > 0 ? -1 : 0) : m.hot) + d * i + n * 4) % n;
                if (!m.items[j].sep && m.items[j].enabled) { m.hot = j; MenuShowItem(j); break; }
            }
            Invalidate();
        };
        // to an end, or a page of items (a long list: the menu key's 21 keys)
        auto jump = [&](int from, int d) {
            const int n = (int)m.items.size();
            for (int j = from; j >= 0 && j < n; j += d)
                if (!m.items[j].sep && m.items[j].enabled) { m.hot = j; MenuShowItem(j); break; }
            Invalidate();
        };
        // its hits gone at once; the owner gets the focus back (UiaFrame says so), its ring showing: the last input was a key
        if (k == VK_ESCAPE) { m.open = false; v.focus = m.owner; v.focusVisible = true; v.hot = 0; Invalidate(); UpdateWindow(v.hwnd); return; }
        if (k == VK_DOWN && !altKey) { step(1); return; }
        if (k == VK_UP && !altKey) { step(-1); return; }
        if (k == VK_TAB) {  // not between its items (a native menu does not): it closes, nothing chosen, and Tab goes on from its control
            m.open = false; v.focus = m.owner; v.hot = 0;
            Invalidate();
            UpdateWindow(v.hwnd);  // its hits gone, the page's back
            MoveFocus(shift);
            return;
        }
        if (k == VK_HOME) { jump(0, 1); return; }
        if (k == VK_END) { jump((int)m.items.size() - 1, -1); return; }
        if (k == VK_NEXT || k == VK_PRIOR) {
            const int page = std::max(1, (int)(MenuRect().h / MenuItemH()) - 1), n = (int)m.items.size();
            const int at = m.hot < 0 ? 0 : m.hot;
            if (k == VK_NEXT) jump(std::min(n - 1, at + page), -1); else jump(std::max(0, at - page), 1);
            return;
        }
        // Return, Space, and as a select's list has it Alt+Up / Alt+Down / F4: the item the keyboard is on
        if ((k == VK_RETURN || k == VK_SPACE || k == VK_F4 || ((k == VK_UP || k == VK_DOWN) && altKey)) && m.hot >= 0) { v.focus = m.owner; v.focusVisible = true; Activate(ID_MENU + m.hot); return; }
        if (k == VK_F4 || ((k == VK_UP || k == VK_DOWN) && altKey)) { m.open = false; v.focus = m.owner; v.focusVisible = true; Invalidate(); return; }
        // type-ahead: the next item whose words start with the letters typed (one letter again: the next one with it)
        if (const wchar_t c = MenuChar(k)) {
            const ULONGLONG now = GetTickCount64();
            if (now - menuTypedAt > 1000) menuTyped.clear();
            menuTypedAt = now;
            menuTyped += c;
            auto starts = [](const wstring& text, const wstring& with) {
                if (text.size() < with.size()) return false;
                for (size_t i = 0; i < with.size(); ++i) if (towlower(text[i]) != with[i]) return false;
                return true;
            };
            const int n = (int)m.items.size();
            for (int pass = 0; pass < 2; ++pass) {
                const wstring want = pass == 0 ? menuTyped : wstring(1, c);
                const int from = want.size() > 1 ? std::max(0, m.hot) : m.hot + 1;
                for (int i = 0; i < n; ++i) {
                    const int j = (from + i + n) % n;
                    const MenuItem& it = m.items[(size_t)j];
                    if (!it.sep && it.enabled && starts(it.text, want)) { m.hot = j; MenuShowItem(j); Invalidate(); return; }
                }
            }
        }
        return;
    }
    // the pages without the mouse: Ctrl+Tab (the other one), Ctrl+1 the Library, Ctrl+2 Settings; the focus on its tab
    if (ctrl && !altKey && (k == VK_TAB || k == '1' || k == '2') && !v.dialog.open && !v.first.open) {
        const int to = k == '1' ? ID_NAV_LIBRARY : k == '2' ? ID_NAV_SETTINGS : v.page == P_SETTINGS ? ID_NAV_LIBRARY : ID_NAV_SETTINGS;
        Activate(to);
        if (OnTabPage(to)) {  // a game's page kept (a cancelled leave question, a refused value): the focus stays put
            if (GetFocus() != v.hwnd) SetFocus(v.hwnd);
            v.focus = to; v.focusVisible = true; Invalidate();
        }
        return;
    }
    // back from a game's page: Alt+Left (as the mouse's Back button)
    if (altKey && k == VK_LEFT) {
        if (v.page == P_GAME && !v.dialog.open && !v.first.open) Navigate(P_LIBRARY);
        return;
    }
    if (altKey) {  // Alt+Down / Alt+Up: a select's list opens; nothing else with Alt
        if ((k == VK_DOWN || k == VK_UP) && !v.dialog.open && !v.first.open)
            for (auto& h : v.hits) if (h.id == v.focus && h.role == R_COMBO && h.enabled) { v.focusVisible = true; Activate(h.id); return; }
        return;
    }
    if (k == VK_TAB) { MoveFocus(shift); return; }
    if (k == VK_ESCAPE) {
        if (v.dialog.open) { v.dialog.result = v.dialog.cancel; v.dialog.open = false; Invalidate(); return; }
        // Escape dismisses what floats over the page (a toast), and never leaves it: Back is Alt+Left, Backspace or the
        // page's Back, as in Windows' Settings and File Explorer (macOS: Cmd+[)
        if (!v.toast.text.empty()) { v.toast = Toast(); Invalidate(); }
        return;
    }
    if (k == VK_RETURN || k == VK_SPACE) {
        // the first run: Return carries on with Continue (its default) unless the keyboard is on another button
        if (k == VK_RETURN && v.first.open && !v.dialog.open) {
            const Hit* on = nullptr;
            for (auto& h : v.hits) if (h.id == v.focus && h.enabled) on = &h;
            if (!on || on->role == R_RADIO) {
                // found first, pressed after the loop: Continue can open a dialog (portable in a folder it cannot write), whose
                // frame builds v.hits again under a loop still walking them
                bool go = false;
                for (auto& h : v.hits) go |= h.id == ID_FR_CONTINUE && h.enabled;
                if (go) { v.focusVisible = true; Activate(ID_FR_CONTINUE); }
                return;
            }
        }
        int press = 0;
        for (auto& h : v.hits) if (h.id == v.focus && h.enabled) { press = h.id; break; }
        if (press) v.focusVisible = true;
        if (k == VK_SPACE) spacePress = press;  // it acts as Space comes up on the same control (SpaceUp), as Windows' buttons do
        else if (press) Activate(press);
        return;
    }
    // Apply on a game's page (the unsaved capsule): Ctrl+S, as Save is everywhere, from any control on the page (ui_uia.cpp)
    if (ctrl && k == 'S') { KeyApply(); return; }
    if (ctrl && k == 'F' && !v.dialog.open && !v.first.open) {
        if (v.page != P_LIBRARY) {  // straight to typing: no entrance for the field to wait out
            if (!Navigate(P_LIBRARY)) return;  // the game's page kept (its unsaved changes)
            v.pageChangedAt = 0;
        }
        v.target[P_LIBRARY] = v.scroll[P_LIBRARY] = 0;
        Invalidate();
        UpdateWindow(v.hwnd);
        if (v.edSearch.h) { SetFocus(v.edSearch.h); SendMessageW(v.edSearch.h, EM_SETSEL, 0, -1); v.focus = ID_SEARCH; }
        return;
    }
    if (v.dialog.open && (k == VK_UP || k == VK_DOWN || k == VK_PRIOR || k == VK_NEXT)) {  // a long text scrolls
        const float step = k == VK_UP ? -40.f : k == VK_DOWN ? 40.f : k == VK_PRIOR ? -160.f : 160.f;
        v.dialog.scroll = std::min(std::max(0.f, v.dialog.scroll + step), v.dialog.maxScroll);
        Invalidate();
        return;
    }
    // a radio group (a segmented control, the first run's choices): every arrow chooses the neighbour, round from one end to
    // the other (Windows' radio groups), and the focus goes with it (UI Automation hears the focus and the choice)
    if (k == VK_LEFT || k == VK_RIGHT || k == VK_UP || k == VK_DOWN) {
        const Hit* on = nullptr;
        for (auto& h : v.hits) if (h.id == v.focus && h.role == R_RADIO && !v.dialog.open) on = &h;
        if (on && repeat) return;  // a held arrow steps once (a choice can save and repaint the window: Appearance, Accent color)
        if (on) {
            auto radio = [](int id) -> const Hit* { for (auto& h : v.hits) if (h.id == id && h.role == R_RADIO) return &h; return nullptr; };
            int lo = on->id, hi = on->id;
            while (radio(lo - 1)) --lo;
            while (radio(hi + 1)) ++hi;
            const int n = hi - lo + 1, d = k == VK_LEFT || k == VK_UP ? -1 : 1;
            for (int i = 1; i < n; ++i) {
                const Hit* to = radio(lo + ((on->id - lo + d * i) % n + n) % n);
                if (!to || !to->enabled) continue;
                const Hit t = *to;
                Activate(t.id);
                FocusHit(&t);
                return;
            }
            return;
        }
    }
    if (v.dialog.open) {  // Left and Right between its buttons
        if (k == VK_LEFT || k == VK_RIGHT) MoveFocus(k == VK_LEFT);
        return;
    }
    if (v.first.open) {  // the card scrolls when the window is too short for it
        const float page = std::max(40.f, (v.first.bodyBottom - v.first.bodyTop) * 0.85f);
        const float d = k == VK_UP ? -40.f : k == VK_DOWN ? 40.f : k == VK_PRIOR ? -page : k == VK_NEXT ? page : k == VK_HOME ? -1e6f : k == VK_END ? 1e6f : 0.f;
        if (d != 0 && v.first.maxScroll > 0) { v.first.scroll = std::min(std::max(0.f, v.first.scroll + d), v.first.maxScroll); Invalidate(); }
        return;
    }
    // the toolbar's tabs: Left and Right go to the other one and open its page
    if ((k == VK_LEFT || k == VK_RIGHT) && (v.focus == ID_NAV_LIBRARY || v.focus == ID_NAV_SETTINGS)) {
        const int to = v.focus == ID_NAV_LIBRARY ? ID_NAV_SETTINGS : ID_NAV_LIBRARY;
        Hit t;
        for (auto& h : v.hits) if (h.id == to) t = h;
        Activate(to);
        if (t.id && OnTabPage(to)) FocusHit(&t);  // not when the game's page was kept (its unsaved changes)
        return;
    }
    // Settings' section index (one Tab stop, Focusables): Up and Down go to the entry above or below, Return or Space opens
    // it (WinUI's NavigationView); at an end the focus stays (a list's arrows do not wrap)
    if ((k == VK_UP || k == VK_DOWN) && v.focus >= ID_S_INDEX0 && v.focus < ID_S_INDEX0 + 3) {
        const int to = v.focus + (k == VK_DOWN ? 1 : -1);
        for (auto& h : v.hits)
            if (h.id == to && h.focusable && h.enabled && to >= ID_S_INDEX0 && to < ID_S_INDEX0 + 3) { const Hit t = h; FocusHit(&t); break; }
        return;
    }
    // a select: Up, Down or F4 open its list (its menu), not the page's scroll (Alt+Down and Alt+Up above)
    if (k == VK_UP || k == VK_DOWN || k == VK_F4)
        for (auto& h : v.hits)
            if (h.id == v.focus && h.role == R_COMBO && h.enabled) { v.focusVisible = true; Activate(h.id); return; }
    if (k == VK_F4) return;
    // a lockup and its action (Install, Update, Repair, Adopt, Play) are one column: Down from the lockup goes to its action,
    // Up from the action back to the lockup (an action that is busy or off is passed over)
    if (v.focusVisible && v.page == P_LIBRARY && (k == VK_UP || k == VK_DOWN)) {
        const bool onTile = v.focus >= ID_TILE && v.focus < ID_TILE + kTileSpan, onAct = v.focus >= ID_TILEQA && v.focus < ID_TILEQA + kTileSpan;
        const int other = k == VK_DOWN && onTile ? v.focus + kTileSpan : k == VK_UP && onAct ? v.focus - kTileSpan : 0;
        if (other)
            for (auto& h : v.hits)
                if (h.id == other && h.focusable && h.enabled) { const Hit t = h; FocusHit(&t); return; }
    }
    // the Library's lockups and its shelf buttons: the arrows walk them (LibraryArrow: Left and Right keep the stop, the
    // lockup or its action; Down goes to the next row's lockup, Up to the row above's action; the lockup when the target
    // has no action that takes the focus). Nothing that way: Left and Right stay, Up and Down scroll the page (below)
    if (v.focusVisible && (k == VK_LEFT || k == VK_RIGHT || k == VK_UP || k == VK_DOWN)) {
        const int to = LibraryArrow(v.focus, (WPARAM)k);
        if (to < 0 && (k == VK_LEFT || k == VK_RIGHT)) return;
        if (to > 0) {
            const bool game = to >= ID_TILE && to < ID_TILEQA + kTileSpan;
            const int tile = game ? ID_TILE + (to - (to >= ID_TILEQA ? ID_TILEQA : ID_TILE)) : 0;
            const Hit *want = nullptr, *alt = nullptr;
            for (auto& h : v.hits) {
                if (!h.focusable || !h.enabled) continue;
                if (h.id == to) want = &h;
                if (tile && h.id == tile) alt = &h;
            }
            if (want || alt) FocusHit(want ? want : alt);
            return;
        }
    }
    switch (k) {
        case VK_DOWN: ScrollBy(56); return;
        case VK_UP: ScrollBy(-56); return;
        case VK_NEXT: ScrollBy(v.H * 0.85f); return;
        case VK_PRIOR: ScrollBy(-v.H * 0.85f); return;
        case VK_HOME: ScrollBy(-1e6f); return;
        case VK_END: ScrollBy(1e6f); return;
        // not while the focus is on a number field whose keys wait for it to scroll in (FocusHit)
        case VK_BACK: if (v.page == P_GAME && !(v.editFocusId && v.editFocusId == v.focus)) Navigate(P_LIBRARY); return;
        default: return;
    }
}

void SetDpi(UINT d) {
    dpi = d;
    ArtScale(d / 96.f);  // art decoded at its drawn size
    MakeFonts();
}
void UpdateSize() {
    RECT rc; GetClientRect(v.hwnd, &rc);
    v.W = rc.right * 96.f / dpi;
    v.H = rc.bottom * 96.f / dpi;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (UiaInvokeMessage(msg, wp, lp)) return 0;
    if (LRESULT r = 0; AccessMessage(msg, wp, lp, r)) return r;  // lone Alt, Browser Back, Space up (ui_uia.cpp)
    switch (msg) {
        case WM_ERASEBKGND: return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps; BeginPaint(hwnd, &ps);
            Advance();
            if (gfx.BeginHwnd(hwnd, dpi)) {
                PaintAll(gfx);
                // the page's height shrank under its scroll (it reports the new one as it paints: a wider window, Show fewer,
                // a game that left): painted again at once at the scroll that fits, never a frame past the content's end
                const Page p = v.page;
                if (v.scroll[p] > std::max(0.f, v.contentH[p] - v.H) + 0.5f || v.target[p] > std::max(0.f, v.contentH[p] - v.H) + 0.5f) PaintAll(gfx);
                // the native fields move to this frame's places before it is presented, so DWM shows them and their painted
                // boxes in the same composition (a glide does not leave a field's text a frame behind its box)
                SyncEdits();
                gfx.EndFrame();
            } else SyncEdits();  // no frame: the fields still follow (hidden where their page is gone)
            EndPaint(hwnd, &ps);
            // no target (the device was lost, or none could be made: no Direct3D 11 device in this session): try again in a
            // quarter second, not in a loop of paints that keeps a core busy
            if (!gfx.rt) SetTimer(hwnd, 3, 250, nullptr);
            // the next frame while something moves (presenting waits for the display, so this runs at its rate); else the
            // clock rests, and the overlay thumb's hold ends on a timer (timer 5) before it fades
            // a window nothing of which is on screen (Present: occluded) does not draw as fast as it can: it looks again on a
            // timer (timer 6) until it shows
            if (gfx.rt && !IsIconic(hwnd) && Animating()) { if (gfx.occluded) SetTimer(hwnd, 6, 200, nullptr); else InvalidateRect(hwnd, nullptr, FALSE); }
            else {
                lastAdvance = 0;
                if (v.contentH[v.page] > v.H + 1 && ThumbAlpha() >= 1 && !Hot(ID_SCROLLBAR) && !v.dragThumb) {
                    const long long left = (long long)kThumbHoldMs - ((long long)NowMs() - (long long)v.scrollSeenAt);
                    if (left > 0) SetTimer(hwnd, 5, (UINT)left + 5, nullptr);
                }
            }
            {  // the focus kept in view: a page change put it on a control (revealFocus), or the unsaved capsule appeared over it
                static float lastBar = 0;
                const float bar = GameBarH();
                if ((revealFocus || (bar > lastBar && v.focusVisible)) && v.focus)
                    for (auto& h : v.hits) if (h.id == v.focus) { Reveal(&h); break; }
                revealFocus = false;
                lastBar = bar;
            }
            UiaFrame();  // what changed in this frame, for UI Automation
            // the page moved under a still pointer (the wheel, PgDn, the smooth scroll, a page change): what is under it now
            // is hot (its hover, its tooltip, its upscale wipe), not what was there before. Only once the content rests (no
            // glide, kHoverRestMs after the last scroll): while it moves the hover stays as it was (timer 8 looks again)
            if (v.mouseX >= 0 && !v.dragThumb && !v.menu.open && !v.down && GetKeyState(VK_LBUTTON) >= 0) {
                const long long rest = (long long)kHoverRestMs - ((long long)NowMs() - (long long)scrollInputAt);
                if (std::fabs(v.scroll[v.page] - v.target[v.page]) > 0.5f) {
                    // gliding: the next frame comes anyway
                } else if (rest > 0) SetTimer(hwnd, 8, (UINT)rest + 5, nullptr);
                else {
                    const Hit* h = HitAt(v.mouseX, v.mouseY);
                    const int hot = h ? h->id : 0;
                    if (hot != v.hot) { v.hot = hot; Invalidate(); Kick(); }
                    else if (WipeArtAt(v.mouseX, v.mouseY) != v.wipeHot) Invalidate();
                }
            }
            UpdateTip();  // the tooltip under a still pointer follows its words (a wait that ended) and its place
            // a page that just opened and scrolls flashes its thumb (macOS; the pointer moving shows it too: WM_MOUSEMOVE)
            {
                static ULONGLONG flashedFor = 1;
                if (v.pageChangedAt != flashedFor) {
                    flashedFor = v.pageChangedAt;
                    if (v.contentH[v.page] > v.H + 1 && !v.first.open) { v.scrollSeenAt = NowMs(); InvalidateRect(hwnd, nullptr, FALSE); }
                }
            }
            return 0;
        }
        case WM_MOVE:
            if (v.live && !IsZoomed(hwnd) && !IsIconic(hwnd)) GetWindowRect(hwnd, &normalRect);
            UpdateTip();  // the keyboard's tooltip moves with the window (ui_frame.cpp)
            break;
        case WM_SIZE:
            if (v.live && wp == SIZE_RESTORED) GetWindowRect(hwnd, &normalRect);
            UpdateSize();
            v.maximized = wp == SIZE_MAXIMIZED;  // the caption's Maximize becomes Restore
            gfx.Resize(hwnd);
            Invalidate();
            if (wp != SIZE_MINIMIZED) UpdateWindow(hwnd);  // painted at the new size now (a live resize does not show a stale frame)
            return 0;
        case WM_ACTIVATE:  // the caption buttons' glyphs dim while another window is active (as Windows draws them)
            v.active = LOWORD(wp) != WA_INACTIVE;
            if (LOWORD(wp) == WA_INACTIVE) {
                // the text field with the keys gets them back when the window is active again (WM_SETFOCUS); an open menu
                // closes, as a native menu does
                const HWND f = GetFocus();
                editFocusBack = f && f != hwnd && GetParent(f) == hwnd ? f : nullptr;
                if (v.menu.open) { v.menu.open = false; v.focus = v.menu.owner; Invalidate(); }
            }
            Invalidate();
            break;  // DefWindowProc gives the window the focus when it is activated (WM_SETFOCUS)
        case WM_NCACTIVATE:  // nothing of the system caption is painted over the toolbar (lParam -1)
            if (v.live) return DefWindowProcW(hwnd, msg, wp, -1);
            break;
        case WM_NCCALCSIZE:  // no system caption: the client area starts at the window's top edge (the toolbar is the title bar)
            if (wp && v.live) {
                auto* p = (NCCALCSIZE_PARAMS*)lp;
                const LONG top = p->rgrc[0].top;
                const LRESULT r = DefWindowProcW(hwnd, msg, wp, lp);  // the side and bottom resize borders as Windows has them
                p->rgrc[0].top = top;
                if (IsZoomed(hwnd)) p->rgrc[0].top += FrameY();  // maximized, the frame's top hangs off the screen
                return r;
            }
            break;
        case WM_NCHITTEST: {  // the toolbar's empty parts move the window; the caption buttons are Windows' own (snap layouts)
            if (!v.live) break;
            const LRESULT d = DefWindowProcW(hwnd, msg, wp, lp);
            if (d != HTCLIENT) return d;
            POINT pt{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
            ScreenToClient(hwnd, &pt);
            RECT rc; GetClientRect(hwnd, &rc);
            if (!IsZoomed(hwnd) && pt.y < FrameY())  // the top resize border, inside the client area
                return pt.x < 2 * FrameX() ? HTTOPLEFT : pt.x >= rc.right - 2 * FrameX() ? HTTOPRIGHT : HTTOP;
            const float x = pt.x * 96.f / dpi, y = pt.y * 96.f / dpi;
            if (y >= kBarH) return HTCLIENT;
            if (const int c = CaptionAt(x, y)) return c == ID_CAP_MIN ? HTMINBUTTON : c == ID_CAP_MAX ? HTMAXBUTTON : HTCLOSE;
            if (const Hit* h = HitAt(x, y); h && (h->role != R_TEXT || v.tips.count(h->id))) return HTCLIENT;  // Library | Settings, a sheet's control, a cut status line's tooltip
            return HTCAPTION;
        }
        case WM_NCMOUSEMOVE: {  // over a caption button: its hover (Windows shows the snap layouts over Maximize itself)
            const int c = wp == HTMINBUTTON ? ID_CAP_MIN : wp == HTMAXBUTTON ? ID_CAP_MAX : wp == HTCLOSE ? ID_CAP_CLOSE : 0;
            if (c != v.hot) { v.hot = c; Invalidate(); UpdateTip(); }
            TRACKMOUSEEVENT t{sizeof t, TME_LEAVE | TME_NONCLIENT, hwnd, 0};
            TrackMouseEvent(&t);
            break;
        }
        case WM_NCMOUSELEAVE:
            if (v.hot >= ID_CAP_MIN && v.hot <= ID_CAP_CLOSE) { v.hot = 0; Invalidate(); }
            if (v.down >= ID_CAP_MIN && v.down <= ID_CAP_CLOSE) { v.down = 0; Invalidate(); }
            break;
        case WM_NCLBUTTONDOWN: case WM_NCLBUTTONDBLCLK: {
            if (v.menu.open) { v.menu.open = false; Invalidate(); }  // a press on the title bar closes an open menu, as anywhere else
            const int c = wp == HTMINBUTTON ? ID_CAP_MIN : wp == HTMAXBUTTON ? ID_CAP_MAX : wp == HTCLOSE ? ID_CAP_CLOSE : 0;
            if (c) { v.down = v.hot = c; Invalidate(); return 0; }  // drawn and carried out here (Windows would draw its own button)
            break;
        }
        case WM_NCLBUTTONUP: {
            const int c = wp == HTMINBUTTON ? ID_CAP_MIN : wp == HTMAXBUTTON ? ID_CAP_MAX : wp == HTCLOSE ? ID_CAP_CLOSE : 0;
            if (v.down >= ID_CAP_MIN && v.down <= ID_CAP_CLOSE) {
                const int down = v.down;
                v.down = 0;
                Invalidate();
                if (c == down) Activate(c);
                return 0;
            }
            break;
        }
        case WM_QUERYENDSESSION:  // Windows shutting down or signing out while an operation writes a game's files: it waits
            // (work that only reads does not hold it up: Collect logs, "Preparing...", the first run's checks: WritesNow)
            if (WritesNow()) { BlockShutdown(true); return FALSE; }
            // unsaved changes on a game's page (the capsule): held up with the reason on Windows' screen, as a document with
            // unsaved changes is (the player can still shut down anyway; a critical shutdown is never held up)
            if (v.live && v.page == P_GAME && Dirty() && !(lp & ENDSESSION_CRITICAL)) {
                ShutdownBlockReasonCreate(hwnd, (L"You have unsaved changes to the settings of " + v.cur.name + L".").c_str());
                return FALSE;
            }
            BlockShutdown(false);  // nothing holds it up any more (changes applied or discarded since a shutdown was held up)
            return TRUE;
        case WM_ENDSESSION:
            // ending: Windows ends the process without WM_DESTROY, so what a close saves is saved here (the window's place, the
            // Library's list for the next start)
            if (wp) { if (v.live && !v.fixture && !replaceAsking) { SavePlacement(hwnd); SaveLibrarySnapshot(); } return 0; }
            if (!WritesNow()) BlockShutdown(false);  // cancelled: the reason goes (an operation's stays while it writes)
            return 0;
        case WM_DPICHANGED: {
            SetDpi(HIWORD(wp));
            const RECT* r = (const RECT*)lp;
            SetWindowPos(hwnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
            SetWindowIcons();  // the taskbar's and Alt+Tab's icons drawn again at this monitor's size (sharp at 200 %)
            if (tipWnd) SendMessageW(tipWnd, TTM_SETMAXTIPWIDTH, 0, Px(480));
            UpdateSize();
            Invalidate();
            return 0;
        }
        case WM_GETMINMAXINFO: {
            auto* mm = (MINMAXINFO*)lp;
            RECT r{0, 0, Px(760), Px(560)};
            AdjustWindowRectExForDpi(&r, WS_OVERLAPPEDWINDOW, FALSE, 0, dpi);
            r.top = 0;  // no caption: the client area starts at the window's top
            mm->ptMinTrackSize = {r.right - r.left, r.bottom - r.top};
            // never larger than the monitor's work area (1280 x 800 at 150 % is 533 DIPs high): the whole window, its first
            // run's Continue and its unsaved capsule stay on the screen; the pages scroll instead
            MONITORINFO mi{sizeof mi};
            if (GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST), &mi)) {
                mm->ptMinTrackSize.x = std::min<LONG>(mm->ptMinTrackSize.x, mi.rcWork.right - mi.rcWork.left);
                mm->ptMinTrackSize.y = std::min<LONG>(mm->ptMinTrackSize.y, mi.rcWork.bottom - mi.rcWork.top);
            }
            return 0;
        }
        case WM_CTLCOLOREDIT: {  // the text fields in the theme's colours (on the painted field: the fill)
            if (!editBrush) editBrush = CreateSolidBrush(Ref(pal.fill));
            HDC dc = (HDC)wp;
            SetTextColor(dc, Ref(pal.text));
            SetBkColor(dc, Ref(pal.fill));
            return (LRESULT)editBrush;
        }
        case WM_SETTINGCHANGE: {  // Windows' light / dark app mode or its accent colour changed: followed when the window follows them
            // Animation effects (SPI_SETCLIENTAREAANIMATION), a Contrast theme, Transparency effects may have changed: read
            // again when next asked; a Contrast theme turned on or off (or another one chosen) is followed at once, whatever
            // the Appearance setting (as WinUI apps do)
            const bool contrastWas = pal.hc;
            MotionRefresh();
            const bool immersive = lp && wcscmp((const wchar_t*)lp, L"ImmersiveColorSet") == 0;
            if (HighContrast() != contrastWas || wp == SPI_SETHIGHCONTRAST || (lp && wcscmp((const wchar_t*)lp, L"WindowsThemeElement") == 0) ||
                (immersive && ((gl.global.appearance != L"light" && gl.global.appearance != L"dark") || gl.global.accent == L"windows")))
                ApplyTheme();
            else if (immersive) Invalidate();  // Transparency effects: the floating material follows
            break;
        }
        case WM_SYSCOLORCHANGE:  // a Contrast theme's colours edited: taken again
            if (pal.hc) ApplyTheme();
            break;
        case WM_DWMCOLORIZATIONCOLORCHANGED:  // the Windows accent colour (Accent color: Windows follows it live)
            if (gl.global.accent == L"windows") ApplyTheme();
            break;
        case WM_COMMAND: {
            const int id = LOWORD(wp), code = HIWORD(wp);
            if (code == EN_CHANGE) {
                wchar_t b[2048] = L"";
                GetWindowTextW((HWND)lp, b, 2048);
                if (id == kEditSearch) {
                    v.search = b; v.scroll[P_LIBRARY] = v.target[P_LIBRARY] = std::min(v.scroll[P_LIBRARY], 0.f);
                    SetTimer(hwnd, 7, 500, nullptr);  // its result said once typing pauses (WM_TIMER 7)
                }
                else if (id == kEditUrl) v.urlText = b;
                else if (id >= kEditOpti0 && id < kEditOpti0 + 64 && (size_t)(id - kEditOpti0) < v.optiValues.size()) {
                    string& val = v.optiValues[id - kEditOpti0];
                    if (val != ToUtf8(b)) { val = ToUtf8(b); lastSettingId = ID_G_OPTI0 + (id - kEditOpti0) * 10; }  // typed (Discard's own reset is no change)
                }
                Invalidate();
            }
            if (code == EN_SETFOCUS) {
                v.focus = id == kEditSearch ? ID_SEARCH : id == kEditUrl ? ID_S_URL : ID_G_OPTI0 + (id - kEditOpti0) * 10;
                Invalidate();
            }
            return 0;
        }
        case WM_MOUSEMOVE: {
            const float x = GET_X_LPARAM(lp) * 96.f / dpi, y = GET_Y_LPARAM(lp) * 96.f / dpi;
            if (v.dragThumb) {
                const Page p = v.page;
                v.scrollSeenAt = NowMs();
                // the thumb's track as DrawScrollbar has it (under the toolbar, between the game page's pinned plate and its
                // unsaved capsule)
                const float ch = v.contentH[p], visible = v.H - kBarH, total = std::max(visible + 1, ch - kBarH);
                const float track = visible - (p == P_GAME ? std::max(0.f, v.pinH - kBarH) + GameBarH() : 0.f);
                const float th = std::max(40.f, track * visible / total);
                const float k = (ch - v.H) / std::max(1.f, track - th);
                v.scroll[p] = v.target[p] = std::min(std::max(0.f, v.dragScroll + (y - v.dragY) * k), std::max(0.f, ch - v.H));
                Invalidate();
                return 0;
            }
            const Hit* h = HitAt(x, y);
            int hot = h ? h->id : 0;
            if (v.menu.open && (x != v.mouseX || y != v.mouseY)) {  // moved: not a move Windows sends when a window shows or hides
                v.menu.hot = -1;
                if (hot >= ID_MENU && hot < ID_MENU + 100) v.menu.hot = hot - ID_MENU;
            }
            if (hot != v.hot) {
                if (v.hot == ID_SCROLLBAR) v.scrollSeenAt = NowMs();  // the thumb stays a moment after the pointer leaves its track
                v.hot = hot; Invalidate(); Kick();
            }
            // the pointer moving over a page that scrolls shows its thumb (Windows 11's overlay scroll bars)
            if (y >= kBarH && v.contentH[v.page] > v.H + 1 && !v.menu.open && !v.dialog.open && !v.first.open && (x != v.mouseX || y != v.mouseY)) {
                if (ThumbAlpha() < 1) Invalidate();
                v.scrollSeenAt = NowMs();
            }
            // the upscale wipe follows the pointer onto art and off it (a lockup's name is the same hit as its art); the
            // toast fades away under the pointer and comes back when it leaves
            const bool inToast = v.toastRc.w > 0 && v.toastRc.has(x, y), wasInToast = v.toastRc.w > 0 && v.toastRc.has(v.mouseX, v.mouseY);
            v.mouseX = x; v.mouseY = y;
            if (WipeArtAt(x, y) != v.wipeHot || inToast != wasInToast) { Invalidate(); Kick(); }
            UpdateTip();
            TRACKMOUSEEVENT t{sizeof t, TME_LEAVE, hwnd, 0};
            TrackMouseEvent(&t);
            return 0;
        }
        case WM_MOUSELEAVE:
            if (!v.dragThumb) {
                if (v.hot == ID_SCROLLBAR) v.scrollSeenAt = NowMs();  // the thumb holds, then fades, as when the pointer leaves its track
                v.mouseX = v.mouseY = -1;
                if (v.hot || v.wipeHot || v.toastRc.w > 0) { v.hot = 0; Invalidate(); Kick(); }
            }
            UpdateTip();
            return 0;
        case WM_SETCURSOR:
            if (LOWORD(lp) == HTCLIENT) {
                if (pageWaiting && !parkedPage && modalDepth == 0) { SetCursor(LoadCursorW(nullptr, IDC_APPSTARTING)); return TRUE; }  // a game's page is being read (OpenGame)
                POINT pt; GetCursorPos(&pt); ScreenToClient(hwnd, &pt);
                const Hit* h = HitAt(pt.x * 96.f / dpi, pt.y * 96.f / dpi);
                SetCursor(LoadCursorW(nullptr, h && h->enabled && h->role != R_EDIT && !Passive(h->role) && h->id != ID_SCROLLBAR ? IDC_HAND : h && h->enabled && h->role == R_EDIT ? IDC_IBEAM : IDC_ARROW));
                return TRUE;
            }
            break;
        case WM_LBUTTONDOWN: case WM_LBUTTONDBLCLK: {
            // the second click of a double click whose first changed the page, closed a dialog or a menu, ended the first run or
            // opened a dialog in place of another: not a press on what is under the pointer now (SHIP-BAR 1: nothing runs twice)
            if (msg == WM_LBUTTONDBLCLK) {
                const ULONGLONG now = GetTickCount64(), at = std::max({std::min(v.pageChangedAt, now), clickGuardAt});
                if (now - at < GetDoubleClickTime() + 100) return 0;
            }
            const float x = GET_X_LPARAM(lp) * 96.f / dpi, y = GET_Y_LPARAM(lp) * 96.f / dpi;
            const Hit* h = HitAt(x, y);
            if (v.menu.open && !(h && h->id >= ID_MENU && h->id < ID_MENU + 100)) {  // a click outside an open menu closes it, and
                v.menu.open = false;                                              // only that (as a native menu: the control
                Invalidate();                                                     // under it is not pressed)
                return 0;
            }
            SetFocus(hwnd);
            v.focusVisible = false;
            if (!(h && h->role == R_EDIT)) v.editFocusId = 0;  // a click elsewhere: a field that waited for its keys (scrolled out) waits no more
            if (h && h->id == ID_SCROLLBAR) {
                v.scrollSeenAt = NowMs();
                // beside the thumb: a page up or down, as a Windows scroll bar's track does
                if (v.thumbRc.h > 0 && (y < v.thumbRc.y || y >= v.thumbRc.b())) { ScrollBy((y < v.thumbRc.y ? -1.f : 1.f) * (v.H - kBarH) * 0.85f); return 0; }
                v.dragThumb = true; v.dragY = y; v.dragScroll = v.scroll[v.page]; SetCapture(hwnd); return 0;
            }
            if (h && h->role == R_EDIT) { if (h->enabled) Activate(h->id); return 0; }  // a locked field (its row disabled): nothing
            v.down = h && h->enabled && !Passive(h->role) ? h->id : 0;  // words and sheets take no press
            if (v.down) { v.focus = v.down; SetCapture(hwnd); }
            Invalidate();
            return 0;
        }
        case WM_LBUTTONUP: {
            if (v.dragThumb) { v.dragThumb = false; ReleaseCapture(); Invalidate(); return 0; }
            const float x = GET_X_LPARAM(lp) * 96.f / dpi, y = GET_Y_LPARAM(lp) * 96.f / dpi;
            const int down = v.down;
            v.down = 0;
            if (GetCapture() == hwnd) ReleaseCapture();
            const Hit* h = HitAt(x, y);
            if (down && h && h->id == down && h->enabled) Activate(down);
            Invalidate();
            return 0;
        }
        case WM_MOUSEWHEEL: {
            // one step everywhere: a notch moves a page, a dialog's text and the first-run card the same distance, the lines (or
            // the screen) Windows' mouse setting says (three lines, 96 DIPs, by default)
            const float notch = WheelNotch(v.dialog.open || v.first.open ? v.H * 0.5f : (v.H - kBarH) * 0.85f);
            const float step = -(float)GET_WHEEL_DELTA_WPARAM(wp) / WHEEL_DELTA * notch;
            if (v.dialog.open) {  // a dialog's long text scrolls
                v.dialog.scroll = std::min(std::max(0.f, v.dialog.scroll + step), v.dialog.maxScroll);
                Invalidate();
                return 0;
            }
            if (v.first.open) {  // the first-run card scrolls when the window is too short for it
                v.first.scroll = std::min(std::max(0.f, v.first.scroll + step), v.first.maxScroll);
                Invalidate();
                return 0;
            }
            if (v.menu.open) {  // over a menu that scrolls (a long list): the menu scrolls; else the wheel closes it
                POINT pt{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
                ScreenToClient(hwnd, &pt);
                if (MenuRect().has(pt.x * 96.f / dpi, pt.y * 96.f / dpi) && MenuScroll(-(float)GET_WHEEL_DELTA_WPARAM(wp) / WHEEL_DELTA * MenuItemH() * 3)) return 0;
                v.menu.open = false;
                Invalidate();
            }
            // a mouse wheel's notches (whole multiples of WHEEL_DELTA) glide; a precision touchpad's small deltas follow the
            // fingers at once (an ease on top of them feels floaty)
            if (GET_WHEEL_DELTA_WPARAM(wp) % WHEEL_DELTA != 0) ScrollNow(step);
            else ScrollBy(step);
            return 0;
        }
        case WM_CAPTURECHANGED:  // the capture taken away (Alt+Tab, the Win key, a dialog opening): no drag or press stays on
            if ((HWND)lp != hwnd && (v.dragThumb || v.down)) { v.dragThumb = false; v.down = 0; Invalidate(); }
            return 0;
        case WM_KEYDOWN: case WM_SYSKEYDOWN:
            if (RrPrepRunningNow() && wp == VK_ESCAPE) { CancelRrPrep(); return 0; }  // Esc cancels Preparing Ray Reconstruction
            // Alt with: Left (back), Down / Up (a select's list); F10 as before; everything else is the system's (Alt+F4)
            if (msg == WM_SYSKEYDOWN && wp != VK_F10 && wp != VK_LEFT && wp != VK_DOWN && wp != VK_UP) break;
            OnKey(wp, lp);
            return 0;
        case WM_XBUTTONUP:  // the mouse's Back button: back from a game's page
            if (GET_XBUTTON_WPARAM(wp) == XBUTTON1 && v.page == P_GAME && !v.dialog.open && !v.first.open && !v.menu.open && modalDepth == 0) Navigate(P_LIBRARY);
            return TRUE;
        case WM_SETFOCUS:
            if (editFocusBack && IsWindow(editFocusBack) && IsWindowVisible(editFocusBack) && !v.dialog.open) SetFocus(editFocusBack);
            editFocusBack = nullptr;
            return 0;
        case WM_TIMER:
            // 1: the Library's scanning skeleton's or a busy indicator's next step (set as it draws); 2: a toast's fade begins; 3: no drawing
            // target yet, try again; 5: the overlay thumb's hold ended (it fades now); 6: a frame while the window is occluded;
            // 8: a scroll came to rest (the hover under the pointer looks again: WM_PAINT); 9: a progress bar's next light step
            if (wp == 1 || wp == 2 || wp == 3 || wp == 5 || wp == 6 || wp == 8 || wp == 9) { KillTimer(hwnd, wp); InvalidateRect(hwnd, nullptr, FALSE); }
            if (wp == 10) {  // 10: a game's page still being read (OpenGame): shown now, its answer opens there
                KillTimer(hwnd, 10);
                const int i = pageAskedIndex;
                if (pageWaiting && pageWaiting == pageSeq && !parkedPage && modalDepth == 0 && !v.first.open && pageAskedOn == P_LIBRARY && v.page == P_LIBRARY &&
                    i >= 0 && i < (int)v.games.size() && IEq(v.games[(size_t)i].dir, pageAsked.dir)) {
                    DropOptiEdits(0);
                    ShowReadingPage(pageAsked, i);
                    pageAskedOn = P_GAME;
                    lastSettingId = 0;
                }
                return 0;
            }
            if (wp == 7) {  // 7: the search's result for Narrator, half a second after the last key: how many games match, or none
                KillTimer(hwnd, 7);
                if (v.page == P_LIBRARY && !v.search.empty() && !v.games.empty() && !v.dialog.open && !v.first.open) {
                    const wstring q = Lower(v.search);
                    int n = 0;
                    for (int i = 0; i < (int)v.games.size() && i < kTileSpan; ++i) n += Lower(v.games[i].name).find(q) != wstring::npos;
                    UiaNotify(n ? std::to_wstring(n) + (n == 1 ? L" game matches" : L" games match") : L"No game matches \u201c" + v.search + L"\u201d");
                }
                return 0;
            }
            if (wp == 4) {  // a running game: still running? Asked on a worker (each check snapshots every process: never on the
                // window's thread, where it could cost a frame); one that closed is read again there too, its page and lockup unlock
                // dir, exe, name, source; rc6: for a game Play started, the time since and whether it was seen running (WatchPlayed)
                std::vector<std::tuple<wstring, wstring, wstring, wstring, ULONGLONG, bool, bool>> running;
                bool starting = false;  // a game Play started moments ago: not looked at yet (its store may start it again: Launching)
                for (auto& e : v.games)
                    if (e.status == GameStatus::Running) {
                        if (Launching(e.exe)) { starting = true; continue; }
                        const auto pl = played.find(Lower(e.exe));
                        const bool byPlay = pl != played.end();
                        running.push_back({e.dir, e.exe, e.name, e.source, byPlay ? GetTickCount64() - pl->second.at : 0, byPlay && pl->second.seen, byPlay});
                    }
                if (running.empty()) { if (!starting) KillTimer(hwnd, 4); return 0; }
                if (pollingRunning) return 0;  // the last check is still running
                pollingRunning = true;
                const Profile defaults = gl.global.defaults;
                Worker([hwnd, running, defaults] {
                    auto* m = new EntriesMsg;
                    m->poll = true;
                    std::shared_lock<std::shared_mutex> opti(optiLock);  // an install may be preparing the OptiScaler package
                    for (auto& c : running) {
                        // with its page's data: what the game changed in its files while it ran shows on its open page (a
                        // throw: looked at again on the next tick, and the result is always posted: pollingRunning ends)
                        try {
                            const bool up = GameRunning(std::get<1>(c));
                            if (std::get<6>(c)) {  // Play started it (rc6): starting until seen running, then closed when it is not
                                const PlayWatch w = WatchPlayed(std::get<4>(c), std::get<5>(c), up);
                                if (w == PlayWatch::Running && !std::get<5>(c)) m->seen.push_back(std::get<1>(c));
                                if (w != PlayWatch::Closed) continue;
                                m->closed.push_back(std::get<1>(c));
                            } else if (up) continue;
                            EntriesMsg::Item it{std::get<0>(c), std::get<1>(c), EntryFor(g.ctx, std::get<1>(c), std::get<2>(c), std::get<3>(c)), PageData()};
                            ReadPageDataSafe(it.fresh, defaults, it.page);
                            m->items.push_back(std::move(it));
                        } catch (...) {}
                    }
                    if (!PostMessageW(hwnd, WM_APP_ENTRIES, 0, (LPARAM)m)) delete m;
                });
            }
            return 0;
        case WM_APP_ENTRIES: {  // games read again off the window's thread (a game that closed)
            std::unique_ptr<EntriesMsg> m((EntriesMsg*)lp);
            if (m->poll) pollingRunning = false;
            for (auto& x : m->seen) if (auto it = played.find(Lower(x)); it != played.end()) it->second.seen = true;  // rc6 (WatchPlayed)
            for (auto& x : m->closed) played.erase(Lower(x));
            if (m->recheck) {
                rechecking = false;
                // Check again: what it found, said (a toast is announced too); nothing read (the folder could not be read): said so.
                // The toast shows wherever the player is by then: it names the game, as the operations' toasts do
                if (m->items.empty()) ShowToast(m->name + L"'s folder couldn't be read. Try again.", 2);
                else {
                    const GameEntry& f = m->items[0].fresh;
                    const wstring tool = f.needs.empty() ? wstring(L"REFramework") : f.needs, game = f.name.empty() ? m->name : f.name;
                    ShowToast(f.needs.empty() || f.needsFound ? tool + (f.hasRecord ? L" is back. WrongGPU can run in " + game + L" again." : L" is in " + game + L". You can install WrongGPU now.")
                                                              : tool + L" isn't in " + game + L"'s folder yet.", f.needs.empty() || f.needsFound ? 1 : 0);
                }
            }
            for (auto& it : m->items) {
                DirChanged(it.dir);  // a page read asked before this is read again (WM_APP_PAGE)
                if (v.scanning) doneSinceScan.push_back({it.dir, it.exe, it.fresh, true});
                // its open page is shown again from the files as the game left them (the outside-change note, OptiScaler's
                // values), unsaved changes kept, as after an operation; an Apply then never writes over what the game wrote unasked
                const bool onPage = v.page == P_GAME && IEq(v.cur.exe, it.exe);
                ReloadEntry(it.dir, it.exe, onPage, &it.fresh, it.page.have ? &it.page : nullptr, !m->again);
            }
            KeepLaunching();
            WatchRunning();
            Invalidate();
            return 0;
        }
        case WM_GETOBJECT: {
            const LRESULT r = UiaGetObject(hwnd, wp, lp);
            if (r) return r;
            break;
        }
        case WM_GETTITLEBARINFOEX: {  // the system title bar's elements (UI Automation: Minimize, Maximize, Close) where the toolbar draws them
            auto* ti = (TITLEBARINFOEX*)lp;
            if (!v.live || !ti || ti->cbSize < sizeof(TITLEBARINFOEX)) break;
            auto screen = [&](float x, float w) {  // a part of the toolbar, in screen pixels
                RECT q{Px(x), 0, Px(x + w), Px(kBarH)};
                MapWindowPoints(hwnd, nullptr, (POINT*)&q, 2);
                return q;
            };
            const float cx = v.W - CaptionW();
            ti->rcTitleBar = screen(0, v.W);
            for (int i = 0; i <= CCHILDREN_TITLEBAR; ++i) { ti->rgrect[i] = RECT{}; ti->rgstate[i] = i == 1 || i == 4 ? STATE_SYSTEM_INVISIBLE : 0; }  // 1 reserved, 4 Help: none
            ti->rgrect[2] = screen(cx, kCapW);              // Minimize
            ti->rgrect[3] = screen(cx + kCapW, kCapW);      // Maximize / Restore
            ti->rgrect[5] = screen(cx + 2 * kCapW, kCapW);  // Close
            return TRUE;  // (0 makes UI Automation guess them from the system metrics)
        }
        case WM_APP_ART: Invalidate(); return 0;
        case WM_APP_PAGE: {  // a game's page read on a worker (OpenGame): it opens now (lParam 0: one parked under a modal loop)
            std::unique_ptr<PageMsg> m((PageMsg*)lp);
            if (!m) m = std::move(parkedPage);
            if (!m || m->seq != pageSeq) return 0;  // a newer ask waits (this one's answer is old)
            if (modalDepth > 0) { parkedPage = std::move(m); return 0; }  // never under a question or inside a file picker's loop
            pageWaiting = 0;
            {  // the pointer stops saying it works
                POINT pt; GetCursorPos(&pt);
                if (WindowFromPoint(pt) == hwnd) SendMessageW(hwnd, WM_SETCURSOR, (WPARAM)hwnd, MAKELPARAM(HTCLIENT, WM_MOUSEMOVE));
            }
            // only where it was asked (the player went to Settings meanwhile: not taken away from there), and only a game the
            // Library still lists (a scan may have landed: its entry as the Library has it now)
            const GameEntry* now = v.page == pageAskedOn && !v.first.open && (!v.pageReading || IEq(v.cur.dir, m->e.dir)) ? EntryOf(m->e.dir, m->e.exe) : nullptr;
            if (!now || !IEq(now->exe, m->e.exe)) {
                if (v.pageReading && v.page == P_GAME && IEq(v.cur.dir, m->e.dir)) Navigate(P_LIBRARY);  // its game is gone
                Invalidate();
                return 0;
            }
            const GameEntry e = *now;
            // never read on the window's thread: the game's entry changed meanwhile (a scan: installed or uninstalled since),
            // so its page is read again on a worker; its files could not be read there, said in words (the page does not
            // open with values that aren't the game's: Apply would write them)
            // The same when an operation on it ended, or it was read again, after the ask (DirChanges): what was read may be
            // the files before it or half way through it
            if ((m->page.have && e.hasRecord != m->e.hasRecord) || DirChanges(e.dir) != m->changes) {
                for (size_t i = 0; i < v.games.size(); ++i) if (&v.games[i] == now) { OpenGame((int)i); break; }
                return 0;
            }
            if (!m->page.have) {
                if (v.pageReading && v.page == P_GAME) Navigate(P_LIBRARY);
                Inform(e.name + L"'s page couldn't be opened.\n\nWrongGPU couldn't read the game's settings files. Try again; if it keeps happening, "
                       L"collect the logs and report it.", kIconError);
                return 0;
            }
            ShowOpened(e, &m->page);
            return 0;
        }
        case WM_APP_PLAYED: {  // Play's answer from Windows (a note when it couldn't start: once no dialog is open, as any note)
            std::unique_ptr<PlayMsg> m((PlayMsg*)lp);
            PlayEnded(m->was, m->ok, m->running, m->gone, m->why);
            return 0;
        }
        case WM_APP_PLAY: {  // "Apply and play" whose Apply ended under a modal loop: the game starts now that it closed
            if (modalDepth > 0 || parkedPlayExe.empty()) return 0;
            const wstring dir = parkedPlayDir, exe = parkedPlayExe;
            parkedPlayDir.clear(); parkedPlayExe.clear();
            if (!v.busy) if (const GameEntry* pe = EntryOf(dir, exe)) { const GameEntry pg = *pe; Play(pg); }
            Invalidate();
            return 0;
        }
        case WM_APP_HEALTH: {
            std::unique_ptr<HealthMsg> m((HealthMsg*)lp);
            TakeModelState(*m);
            if (m->gpuDone) SetGpu(m->gpu);
            HealthEnded();  // the next check, if one was asked meanwhile
            Invalidate();
            return 0;
        }
        case WM_APP_UPDATE: {  // wParam 1: "Check now" (a toast); 0: the start-up check (a note when there is a newer release)
            std::unique_ptr<UpdateInfo> up((UpdateInfo*)lp);
            if (wp == 1) v.checkingUpdate = false;
            if (up->newer) { pendingUpdatePage = up->page; InformLater(up->message, kNoteUpdate); }  // a note with a way to the release page
            else if (wp == 1) ShowToast(up->message, up->checked ? 1 : 2);
            Invalidate();
            return 0;
        }
        case WM_APP_NOTE:  // the notes that arrived on their own, one at a time once no dialog is open
            if (lp) { std::unique_ptr<Note> n((Note*)lp); pendingNotes.push_back(*n); }
            if (modalDepth > 0 || pendingNotes.empty()) return 0;
            {
                const Note n = pendingNotes.front();
                pendingNotes.erase(pendingNotes.begin());
                if (n.icon == kNoteUpdate) ShowUpdateNote(n.text, pendingUpdatePage);
                else InformDetails(n.text, n.icon, n.details);
            }
            if (!pendingNotes.empty() && modalDepth == 0) PostMessageW(hwnd, WM_APP_NOTE, 0, 0);
            else if (pendingNotes.empty() && closeAfterNotes) { closeAfterNotes = false; PostMessageW(hwnd, WM_CLOSE, 0, 0); }  // the last note read: Close's promise kept
            return 0;
        case WM_APP_UNINSTALL:  // started again from Apps & features while this window is open: its uninstall question here
            if (!v.live) return 0;
            if (wp == 1 && !uninstallWaiting) return 0;  // the waiting one, asked already (a Close came first: WM_CLOSE asked it)
            if (v.busy || firstRunBusy || modalDepth > 0) {  // it waits and says so; it starts once that ends (UninstallIfWaiting)
                if (!uninstallWaiting)
                    ShowToast(modalDepth > 0 && !v.busy && !firstRunBusy ? L"Uninstall starts once you answer this question."
                              : firstRunBusy ? L"Uninstall starts once WrongGPU finishes checking this PC."
                                             : L"Uninstall starts once WrongGPU finishes what it's doing.", 0);
                uninstallWaiting = true;
                return 0;
            }
            uninstallWaiting = false;
            if (Navigate(P_SETTINGS)) { Invalidate(); Activate(ID_S_SELFUNINSTALL); }
            return 0;
        case WM_APP_TASK: {  // a background task ended: the window does what it left to do (lParam 0: a parked one, ModalClosed)
            std::unique_ptr<TaskMsg> m((TaskMsg*)lp);
            if (modalDepth > 0) {  // a file dialog or a question is open: it ends once that closes (still busy until then)
                if (m) parkedTasks.push_back(std::move(m));
                return 0;
            }
            if (!m) {
                if (parkedTasks.empty()) return 0;
                m = std::move(parkedTasks.front());
                parkedTasks.erase(parkedTasks.begin());
            }
            v.busy = false; v.task = T_NONE;
            BlockShutdown(false);
            if (m->done) m->done();
            UninstallIfWaiting();
            Invalidate();
            return 0;
        }
        case WM_APP_SCAN: {
            std::unique_ptr<ScanMsg> m((ScanMsg*)lp);
            if (wp == 1) {  // the start-up scan's games with WrongGPU, read first: shown while it goes on (still scanning)
                if (m && v.scanning && v.games.empty()) { v.games = std::move(m->games); Invalidate(); }
                return 0;
            }
            v.scanning = false;
            if (!m) {  // the scan failed (logged): the Library keeps what it lists, and says so once. The last session's list
                // (LoadLibrarySnapshot) stays locked: it never held leftovers, coexistence or found files, so no action runs on
                // it until a scan has read the games (BusyWhy: "Rescan..."; the toast asks for it)
                doneSinceScan.clear();
                // the settings changed meanwhile: once more. Add game / Add folder's result waits for that scan (it is the
                // one that lists the added game, and takes a file that is not a game out of launcher.ini again)
                if (v.rescanPending) { v.rescanPending = false; StartScan(); }
                else {
                    // Add game's file waited for this scan to say whether it is a game: unconfirmed, it goes out of launcher.ini
                    // again (no later scan checks it, and no UI could remove a file that is not a game), and the toast says so
                    const bool undone = !addedExe.empty();
                    if (undone) DropAddedExe();
                    addedExe.clear(); addedFolder.clear(); rescanAsked = false;
                    ShowToast(undone ? L"WrongGPU couldn't finish looking for games, so the file you chose isn't added. Try Add game again; if it keeps happening, collect the logs and report it."
                                     : L"WrongGPU couldn't finish looking for games. Try Rescan; if it keeps happening, collect the logs and report it.", 2);
                }
                WatchRunning();
                Invalidate();
                return 0;
            }
            v.listCached = false;  // the last session's list (LoadLibrarySnapshot) ends here: the games are read
            v.wipeAt.clear(); v.wipeOut.clear();  // the lockups' ids are their places: a hover after the scan fades in again
            // the keyboard's lockup by its game, not by its place (a scan can add or reorder games), also the one kept under
            // an open dialog (a scan that lands while a question is open)
            HoldPlaceFocus();
            std::set<wstring> before;  // the games listed before it (Rescan says what it found that is new)
            if (rescanAsked) for (auto& e : v.games) before.insert(Lower(e.dir) + L"|" + Lower(e.exe));
            std::vector<wstring> order;  // the games by place before it: a press or a hover on a lockup stays where its game stays
            for (auto& e : v.games) order.push_back(Lower(e.dir) + L"|" + Lower(e.exe));
            v.games = std::move(m->games);
            // a game added by hand and removed from the Library while the scan ran (the scan read the list before): out again
            v.games.erase(std::remove_if(v.games.begin(), v.games.end(), [](const GameEntry& x) {
                if (x.source != L"Added" || x.hasRecord) return false;
                for (auto& a : gl.global.addedGames) if (IEq(FullPath(a), FullPath(x.exe))) return false;
                return true;
            }), v.games.end());
            // the launcher opens (or a rescan): an Engine.ini choice made before a game's first start, its folder there now (r2b #1)
            for (auto& e : v.games) if (e.rrConfigDue) ApplyRrConfigWhenDue(e.dir, e.exe);
            // games an operation finished on while the scan ran: their entries as it left them (newer than the scan's)
            for (auto& d : doneSinceScan) ReloadEntry(d.dir, d.exe, false, d.have ? &d.fresh : nullptr);
            doneSinceScan.clear();
            if (v.page == P_GAME) {  // the page's game again, from the new list: by its exe, else its folder (as ReloadEntry)
                v.curIndex = -1;
                for (size_t i = 0; i < v.games.size() && v.curIndex < 0; ++i) if (!v.cur.exe.empty() && IEq(v.games[i].exe, v.cur.exe)) v.curIndex = (int)i;
                for (size_t i = 0; i < v.games.size() && v.curIndex < 0; ++i) if (IEq(v.games[i].dir, v.cur.dir)) v.curIndex = (int)i;
                if (v.curIndex >= 0) v.cur = v.games[(size_t)v.curIndex];
                else {  // its folder is gone (moved, uninstalled): the Library, said in words; changes for it cannot be written any more
                    const wstring name = v.cur.name;
                    v.menu.open = false;
                    v.edit = v.saved;
                    v.page = P_LIBRARY; v.pageChangedAt = NowMs(); v.wipeAt.clear(); v.editFocusId = 0;
                    ShowToast(name + L" left the Library: its folder was moved or removed.", 0);
                }
            }
            RestorePlaceFocus();  // gone: the nearest stop (PaintAll)
            {  // only presses and hovers on places whose game moved are let go (a press on a tab, the toolbar or a page stays)
                bool moved = order.size() != v.games.size();
                for (size_t i = 0; i < v.games.size() && !moved; ++i) moved = order[i] != Lower(v.games[i].dir) + L"|" + Lower(v.games[i].exe);
                if (moved) ForgetPlaces();
            }
            KeepLaunching();  // a game Play started moments ago stays running while its store starts it again
            WatchRunning();  // a game started outside the launcher: its page unlocks once it closes
            SaveLibrarySnapshot();  // the next start shows this list at once
            PostMessageW(hwnd, WM_APP_RRPREP, 0, 0);  // 0.6.0 polish: RR prepared again after a driver / WrongGPU update (ui_rrprep.cpp)
            if (v.rescanPending) { v.rescanPending = false; StartScan(); }  // the settings changed while it ran
            else if (!addedExe.empty() || !addedFolder.empty()) {  // Add game / Add a folder: where it went, in words
                if (!addedExe.empty()) {
                    const GameEntry* a = EntryOf(Parent(addedExe), addedExe);
                    const bool listed = a && IEq(a->exe, addedExe);
                    if (!listed) DropAddedExe();  // not a game: out of launcher.ini again (no scan checks it any more; no UI could remove it)
                    ShowToast(listed ? a->name + L" is in the Library."
                                     : L"That file isn't a game WrongGPU can use. Choose the game's own program (not its launcher or setup).",
                              listed ? 1 : 0);
                } else {
                    size_t n = 0;
                    const wstring f = Lower(addedFolder) + (addedFolder.empty() || addedFolder.back() == L'\\' ? L"" : L"\\");
                    for (auto& e : v.games) n += Lower(e.dir).rfind(f, 0) == 0;
                    const wstring where = BaseName(addedFolder).empty() ? addedFolder : BaseName(addedFolder);
                    // one sentence form for every count; none found says what to do next (a game the scan doesn't know: its program)
                    ShowToast(n == 0 ? L"No games found in " + where + L". To add one game, choose its program with Add game… in the Library." : n == 1 ? L"1 game found in " + where + L"." : std::to_wstring(n) + L" games found in " + where + L".",
                              n ? 1 : 0);
                }
                addedExe.clear(); addedFolder.clear();
                rescanAsked = false;
            } else if (rescanAsked) {  // Rescan: its outcome for Narrator (the Library's own line shows it on screen)
                rescanAsked = false;
                std::vector<wstring> found;
                for (auto& e : v.games) if (!before.count(Lower(e.dir) + L"|" + Lower(e.exe))) found.push_back(e.name);
                const size_t all = v.games.size();
                UiaNotify(found.empty() ? L"No new games found. The Library has " + std::to_wstring(all) + (all == 1 ? L" game." : L" games.")
                          : found.size() == 1 ? L"1 new game found: " + found[0] + L"."
                                              : std::to_wstring(found.size()) + L" new games found.",
                          false, 1);
            }
            Invalidate();
            return 0;
        }
        case WM_APP_FIRSTRUN: {
            std::unique_ptr<FirstRunMsg> m((FirstRunMsg*)lp);
            if (wp == 1) {  // the checks so far: the sheet fills in its rows while the rest runs
                if (v.first.busy) { v.first.lines = std::move(m->lines); Invalidate(); }
                return 0;
            }
            firstRunBusy = false;
            v.first.busy = false;
            v.first.lines = std::move(m->lines);
            firstRunModel = m->model;  // carried out on Continue
            firstRunAdopt = m->adopt;
            {  // the checks' outcome for Narrator, in the sheet's own words ("1 needs a look", "6 checks passed")
                int passed = 0, problems = 0;
                for (auto& l : v.first.lines) { passed += l.first == 0; problems += l.first == 2; }
                UiaNotify(problems ? L"This PC: " + std::to_wstring(problems) + (problems == 1 ? L" needs a look" : L" need a look")
                                   : L"This PC: " + std::to_wstring(passed) + (passed == 1 ? L" check passed" : L" checks passed"));
            }
            // the checks are done: Continue (the sheet's default) has the keyboard, unless it is on one of the sheet's own
            // buttons already (Search your games, which checks again)
            if (v.first.open && !(v.focus >= ID_FR_MODEL && v.focus <= ID_FR_SEARCH)) v.focus = ID_FR_CONTINUE;
            v.games = std::move(m->games);
            v.scanning = false;
            TakeModelState(m->health);
            WatchRunning();
            if (v.rescanPending) { v.rescanPending = false; StartScan(); }
            UninstallIfWaiting();
            Invalidate();
            return 0;
        }
        case WM_APP_RRPREP: RunRrPrepQueue(); return 0;
        case WM_APP_RRBG: RrBackgroundMessage(wp); return 0;
        case WM_APP_STEP: {
            std::unique_ptr<StepMsg> m((StepMsg*)lp);
            v.step = StepWords(m->text, m->done, m->total);
            if (m->total > 0) v.progress = (float)m->done / m->total;
            Invalidate();
            return 0;
        }
        case WM_APP_OPDONE: {
            std::unique_ptr<OpDoneMsg> m((OpDoneMsg*)lp);
            if (m->needsPermission) {
                // the window's own Engine.ini apply, not run: the game's folder needs administrator permission (r2c). No
                // result to tell (no toast, no note); the page keeps what it showed, and its card says "Ready to add"
                v.busy = false; v.busyDir.clear(); v.step.clear(); v.busyVerb.clear();
                BlockShutdown(false);
                if (m->op == Op::Apply && IEq(v.pendingApply.dir, m->dir)) {
                    if (v.page == P_GAME && IEq(v.cur.dir, m->dir)) { v.edit = v.pendingApply.edit; v.saved = v.pendingApply.saved; }
                    v.pendingApply = View::PendingApply();
                }
                DirChanged(m->dir);
                if (v.scanning) doneSinceScan.push_back({m->dir, m->exe, m->fresh, m->haveEntry});
                ReloadEntry(m->dir, m->exe, true, m->haveEntry ? &m->fresh : nullptr, m->haveEntry ? &m->page : nullptr);
                RrConfigNeedsClick(m->dir);
                PostMessageW(hwnd, WM_APP_RRPREP, 0, 0);
                Invalidate();
                return 0;
            }
            // Apply and play: the game starts once its settings are written (not after a failure: its note says why)
            const wstring playNow = m->op == Op::Apply && !m->res.code && !playAfterApply.empty() ? playAfterApply : wstring();
            if (m->op == Op::Apply) playAfterApply.clear();
            const int as = IEq(resultAsDir, m->dir) ? resultAs : RES_PLAIN;  // the control's own result words (SayResultAs), this operation only
            resultAs = RES_PLAIN; resultAsDir.clear();
            v.busy = false;
            v.busyDir.clear();
            v.step.clear();
            v.busyVerb.clear();  // a later task's words never name this operation (BusyDoneWords, BusyWhy)
            BlockShutdown(false);
            const wstring name = !m->name.empty() ? m->name : m->haveEntry && !m->fresh.name.empty() ? m->fresh.name : BaseName(m->dir);
            // the warnings: in plain words for the player (the same one once), the core's own behind Show details
            std::vector<wstring> plain;
            wstring raw;
            for (auto& w : m->res.warnings) {
                raw += (raw.empty() ? L"" : L"\n\n") + w;
                const wstring p = PlainWarning(w);
                if (!p.empty() && std::find(plain.begin(), plain.end(), p) == plain.end()) plain.push_back(p);
            }
            const bool queued = v.queueTotal > 0;
            if (queued) ++v.queueDone;
            const int queueTotal = v.queueTotal, queueDone = v.queueDone;  // Update all: its size, and the games done with this one
            const bool onPage = v.page == P_GAME && IEq(v.cur.dir, m->dir);
            const bool applyKept = m->op == Op::Apply && m->res.code && IEq(v.pendingApply.dir, m->dir);
            // a failure: a note (once no dialog is open: it never answers one) titled with the action and the game, the cause
            // and what to do in words, the core's text behind Show details; and a callout on that game's page until it is
            // dismissed. A success: a toast, wherever the player is (one pattern for it); with warnings, a note.
            wstring failBody;  // said once the entry is read again (is the game still listed?)
            if (m->res.code) {
                const int skipped = (int)v.queue.size();
                v.queue.clear(); v.queueTotal = v.queueDone = 0;
                wstring body = PlainCause(m->res.message, m->res.code);
                // Update all stopped here: the games it updated before this one said too (they had no toast of their own)
                if (queued && queueDone > 1) body += queueDone == 2 ? L" The game before it was updated." : L" The " + std::to_wstring(queueDone - 1) + L" games before it were updated.";
                if (skipped) body += skipped == 1 ? L" The other game wasn't updated." : L" The other " + std::to_wstring(skipped) + L" games weren't updated.";
                if (queued) { const wstring left = QueueSkippedWords(); if (!left.empty()) body += L" " + left; }
                batchedResults.clear();
                failBody = body;
            }
            DirChanged(m->dir);  // a page read asked before this ended is read again (WM_APP_PAGE)
            // Update all: taken as each game's turn comes (their folders changed now: a page read of one is read again)
            if (!m->res.code) for (auto& b : m->batched) { batchedResults[Lower(b.first)] = b.second; DirChanged(b.first); }
            if (v.scanning) doneSinceScan.push_back({m->dir, m->exe, m->fresh, m->haveEntry});  // the running scan's result is older than this
            ReloadEntry(m->dir, m->exe, true, m->haveEntry ? &m->fresh : nullptr, m->haveEntry ? &m->page : nullptr);
            // RightGPU: an install, repair or update that worked, on the game's page: the mark flips there and then (BrandNow)
            if (!m->res.code && (m->op == Op::Install || m->op == Op::Repair || m->op == Op::Update) && v.page == P_GAME && IEq(v.cur.dir, m->dir) && RightGpu(v.cur)) {
                v.flipDir = m->dir;
                v.flipAt = NowMs();
            }
            // 0.6.0: RR's kernels prepared next (ui_rrprep.cpp)
            // rc5: not while the game's RR waits for its experimental opt-in, nor where its menu has no RR option (gamerules.h)
            const bool rrHeld = m->haveEntry && (m->fresh.rrHeldOff || m->fresh.rrNotOffered);
            if (!m->res.code && (m->op == Op::Install || m->op == Op::Update || m->op == Op::Repair) && RrGameInstalled(m->dir) && !rrHeld) {
                QueueRrPrep(m->dir, m->exe);
            }
            if (!m->res.code && (m->op == Op::Install || m->op == Op::Update || m->op == Op::Repair) && m->page.have) AskRrConfigLater(m->dir, m->haveEntry ? m->fresh.exe : m->exe, m->page.saved);
            PostMessageW(hwnd, WM_APP_RRPREP, 0, 0);  // and a background re-prepare the operation stopped goes on
            // a game whose folder is gone left the Library (and its page): nothing is promised on it or kept for it
            const bool listed = EntryOf(m->dir, m->exe) != nullptr;
            if (m->res.code) {
                if (applyKept && PendingEdits())
                    failBody += !listed ? L" " + name + L" isn't in its folder any more, so your changes couldn't be kept."
                              : onPage && v.page == P_GAME && IEq(v.cur.dir, m->dir) ? L" Your changes are still on the page."
                                                                                       : L" Your changes are kept: they're back when you open " + name + L".";
                NoteLater(OpFailTitle(m->op, name, as) + L"\n\n" + failBody, kIconError, m->res.message + (raw.empty() ? L"" : L"\n\n" + raw));
            }
            // an Apply that failed: the values the player set come back as unsaved changes (the files kept theirs); on its
            // page now, or when its page opens again (the player left it with Apply)
            if (applyKept && listed) {
                if (v.page == P_GAME && IEq(v.cur.dir, m->dir)) {
                    RestoreEdits(v.pendingApply.edit, v.pendingApply.saved, v.pendingApply.rows, v.pendingApply.opti);
                    for (size_t i = 0; i < v.edOpti.size() && i < v.optiValues.size(); ++i)
                        if (v.edOpti[i].h) SetWindowTextW(v.edOpti[i].h, FromUtf8(v.optiValues[i]).c_str());
                } else keptEdits[Lower(m->dir)] = v.pendingApply;
            }
            if (m->op == Op::Apply) v.pendingApply = View::PendingApply();
            if (m->res.code && listed) { v.notice = OpFailLead(m->op, as) + L" " + PlainCause(m->res.message, m->res.code); v.noticeDir = m->dir; v.noticeTone = 2; }
            else {
                if (IEq(v.noticeDir, m->dir)) v.notice.clear();
                const wstring text = OpSuccess(m->op, m->res, m->req, m->haveEntry ? &m->fresh : nullptr, name, as);
                // Update all of more than one game: while it goes on, each game's result is on its lockup ("Updated to 0.5.5")
                // and spoken; at its end one toast says the whole action is done (a toast per game would replace the last)
                const bool more = queued && !v.queue.empty(), end = queued && v.queue.empty() && queueTotal > 1;
                const wstring ver = g.ctx.data ? wstring(g.ctx.data->version) : wstring();
                const wstring all = std::to_wstring(queueDone) + L" games are updated to WrongGPU " + ver + L".";
                // Update all's end: the games it promised but left out (gone since the scan, not readable), named
                const wstring left = queued && v.queue.empty() ? QueueSkippedWords() : wstring();
                if (plain.empty()) {
                    if (more) UiaNotify(text, false, 1);
                    else if (!left.empty()) NoteLater((end ? all : text) + L"\n\n" + left, 1, L"");
                    else ShowToast(end ? all : text, 1);
                } else {  // nothing fails silently: what else happened, in words
                    wstring body;
                    for (auto& p : plain) body += (body.empty() ? L"" : L" ") + p;
                    if (!end && !left.empty()) body += L" " + left;
                    NoteLater(text + L"\n\n" + body, 2, raw);
                    if (end && !left.empty()) NoteLater(all + L"\n\n" + left, 1, L"");
                    else if (end) ShowToast(all, 1);
                }
            }
            WatchRunning();
            StartHealth(false);  // an install can convert the model, the last uninstall removes it: the status line follows
            if (queued && !m->res.code && !v.queue.empty()) StartNextQueued();  // only the queue's own operations go on with it
            else { v.queue.clear(); v.queueTotal = v.queueDone = 0; if (queued) { queueSkipped.clear(); batchedResults.clear(); } }
            // closed with Apply chosen for unsaved changes: the window closes now that it is written (a failure keeps it open,
            // with the note and the changes back)
            // (notes waiting to be read: once the last of them is closed, WM_APP_NOTE; a WM_CLOSE now would cancel the first)
            if (v.closeAfterOp && !v.busy) {
                v.closeAfterOp = false;
                if (!m->res.code) { if (pendingNotes.empty()) PostMessageW(hwnd, WM_CLOSE, 0, 0); else closeAfterNotes = true; }
            }
            if (!playNow.empty() && !v.busy) {
                // a file picker or a question open (this arrived in its loop): the game starts once it closes, so a note that it
                // couldn't start never opens inside that loop (WM_APP_PLAY)
                if (modalDepth > 0) { parkedPlayDir = m->dir; parkedPlayExe = playNow; }
                else if (const GameEntry* pe = EntryOf(m->dir, playNow)) { const GameEntry pg = *pe; Play(pg); }
            }
            UninstallIfWaiting();
            Invalidate();
            return 0;
        }
        case WM_CLOSE:
            // a question open: Close answers it as Esc does (its safe answer), as a native dialog's owner would not close under
            // it; the next Close closes the window. A file dialog (its own window) keeps the window.
            if (v.dialog.open) { v.dialog.result = v.dialog.cancel; v.dialog.open = false; Invalidate(); return 0; }
            if (modalDepth > 0) { FlashWindow(hwnd, TRUE); return 0; }
            // not half way through a game's files, a conversion, or the first run's adoptions
            if (firstRunBusy) { ShowToast(L"Still checking this PC. The launcher can close once this is done.", 0); return 0; }
            if (v.busy) {  // what it waits for, in words
                const wstring what = v.task == T_COLLECT ? L"Still collecting logs." : v.task == T_MODEL ? L"Still converting the Model M data."
                                     : v.task == T_RRDATA ? L"Still converting the Ray Reconstruction data."
                                     : v.task == T_SELFINSTALL || (v.task == T_FIRSTRUN && v.first.installOffer && v.first.choice == 1) ? L"Still installing WrongGPU for this user."
                                     : v.task == T_FIRSTRUN ? L"Still setting up WrongGPU."
                                     : v.task == T_SELFUNINSTALL ? L"Still uninstalling the launcher."
                                     : LookingAtGames() ? L"Still looking at your games."  // the launcher's uninstall, before it asks
                                     : inspecting && v.busyDir.empty() && v.busyVerb == L"Removing…" ? L"Still removing WrongGPU's data."  // Remove data...
                                     : inspecting ? L"Still reading the game's files." : L"";  // "Preparing...": its questions come next
                if (what.empty()) {  // a game's operation: named, and the window closes by itself once it ends well (a failure's
                    // note stays to be read: WM_APP_OPDONE); Update all: once the last game is done
                    v.closeAfterOp = true;
                    ShowToast(v.queue.empty() ? L"WrongGPU closes once " + BusyDoneWords() + L"." : wstring(L"WrongGPU closes once the updates are done."), 0);
                    return 0;
                }
                ShowToast(what + L" The launcher can close once this is done.", 0);
                return 0;
            }
            if (!LeaveGamePage()) return 0;  // unsaved changes on a game's page: Apply, Discard or stay
            if (v.busy) {  // Apply chosen: it is writing the game's files; the window closes once it is done (WM_APP_OPDONE)
                v.closeAfterOp = true;
                ShowToast(L"Applying your changes. The launcher closes once they're saved.", 0);
                return 0;
            }
            if (uninstallWaiting) {  // Apps & features' Uninstall waited for the work that just ended: asked now, never dropped
                uninstallWaiting = false;
                if (Navigate(P_SETTINGS)) { Invalidate(); Activate(ID_S_SELFUNINSTALL); }
                if (v.busy || !IsWindow(hwnd)) return 0;  // uninstalling: the window closes once that is done
            }
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            StopRrBackground();  // the helper ends at once (what it compiled stays with the driver)
            if (editBrush) { DeleteObject(editBrush); editBrush = nullptr; }
            if (v.live && !v.fixture && !replaceAsking) { SavePlacement(hwnd); SaveLibrarySnapshot(); }  // the list as this session leaves it
            if (v.live) {
                KillTimer(hwnd, 4);
                ShutdownBlockReasonDestroy(hwnd);
            }
            UiaShutdown(hwnd);
            ArtStop();
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

HWND CreateLauncherWindow(bool live) {
    v.live = live;
    v.rrBuild = RrPayloadAvailable(g.ctx);  // this build has the RR files: the RR rows and sections can show (rr-notes.md)
    v.rrEdition = true;  // WrongGPU's edition of OptiScaler (some games' RR needs it): downloaded (optiedition.h)
    SetAccent(AccentFromKey(gl.global.accent), WindowsAccentRgb());
    SetTheme(ThemeIsDark(gl.global.appearance));
    gfx.Init();
    WNDCLASSEXW wc{}; wc.cbSize = sizeof(wc);
    wc.style = CS_DBLCLKS;
    wc.lpfnWndProc = WndProc; wc.hInstance = g.hinst; wc.lpszClassName = L"WrongGPULauncher";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW); wc.hIcon = LoadIconW(g.hinst, MAKEINTRESOURCEW(1)); wc.hIconSm = wc.hIcon;
    RegisterClassExW(&wc);
    if (!live) {  // ui-dump / ui-render: a message-only window (never visible) receives the workers' results
        v.hwnd = CreateWindowExW(0, wc.lpszClassName, L"WrongGPU", 0, 0, 0, 0, 0, HWND_MESSAGE, nullptr, g.hinst, nullptr);
        return v.hwnd;
    }
    // a standard frame (resize borders, snap, the window menu, the taskbar's minimize) whose caption is the app's toolbar:
    // WM_NCCALCSIZE gives the caption's room to the client area, WM_NCHITTEST makes the toolbar's empty parts the title bar
    const DWORD style = WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN;
    // where it was last time (SavedPlacement), on the monitor it was on and at its DPI (no DPI change follows); else the
    // size for the main monitor, centred
    RECT saved{};
    HMONITOR savedMon = nullptr;
    UINT savedDpi = 0;
    bool savedMax = false;
    if (SavedPlacement(saved, savedMon, savedDpi, savedMax)) {
        dpi = savedDpi;
        v.hwnd = CreateWindowExW(0, wc.lpszClassName, L"WrongGPU", style, saved.left, saved.top, saved.right - saved.left, saved.bottom - saved.top, nullptr, nullptr,
                                 g.hinst, nullptr);
        normalRect = saved;
    } else {
        // the size for the monitor the window opens on
        POINT center{GetSystemMetrics(SM_CXSCREEN) / 2, GetSystemMetrics(SM_CYSCREEN) / 2};
        HMONITOR mon = MonitorFromPoint(center, MONITOR_DEFAULTTOPRIMARY);
        UINT mdpi = GetDpiForSystem();
        {
            using Fn = HRESULT(WINAPI*)(HMONITOR, int, UINT*, UINT*);
            if (HMODULE sh = LoadLibraryExW(L"shcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32))
                if (auto f = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(sh, "GetDpiForMonitor")))) { UINT y = 0; f(mon, 0, &mdpi, &y); }
        }
        dpi = mdpi;
        MONITORINFO mi{sizeof mi}; GetMonitorInfoW(mon, &mi);
        const int workW = mi.rcWork.right - mi.rcWork.left, workH = mi.rcWork.bottom - mi.rcWork.top;
        RECT r{0, 0, std::min(Px(kDefaultW), workW - 40), std::min(Px(kDefaultH), workH - 60)};
        AdjustWindowRectExForDpi(&r, style, FALSE, 0, dpi);
        r.top = 0;  // no caption: the client area starts at the window's top
        const int ww = r.right - r.left, wh = r.bottom - r.top;
        v.hwnd = CreateWindowExW(0, wc.lpszClassName, L"WrongGPU", style, mi.rcWork.left + (workW - ww) / 2, mi.rcWork.top + (workH - wh) / 2, ww, wh, nullptr, nullptr, g.hinst, nullptr);
    }
    // the way it starts: maximized as it closed; the shortcut's "Run: Maximized / Minimized" (the start-up info) wins
    startShow = savedMax ? SW_SHOWMAXIMIZED : SW_SHOWNORMAL;
    {
        STARTUPINFOW si{sizeof si};
        GetStartupInfoW(&si);
        if (si.dwFlags & STARTF_USESHOWWINDOW) {
            const WORD s = si.wShowWindow;
            if (s == SW_SHOWMAXIMIZED) startShow = SW_SHOWMAXIMIZED;
            else if (s == SW_SHOWMINIMIZED || s == SW_MINIMIZE || s == SW_SHOWMINNOACTIVE || s == SW_FORCEMINIMIZE) startShow = SW_SHOWMINNOACTIVE;
        }
    }
    dpi = GetDpiForWindow(v.hwnd);
    ArtScale(dpi / 96.f);  // art decoded at its drawn size from the first frame (WM_DPICHANGED comes only when the DPI changes)
    SetWindowPos(v.hwnd, nullptr, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);  // WM_NCCALCSIZE now
    MakeFonts();
    ThemeTitleBar();
    SetWindowIcons();  // the mark, as the toolbar draws it
    UpdateSize();
    ArtStart(v.hwnd, WM_APP_ART);
    return v.hwnd;
}

void OpenMoreMenu() {
    Activate(ID_G_MORE);
}

}  // namespace win

int RunLauncherWindow(bool uninstall) {
    // A new release started from wherever it was downloaded offers to replace the installed launcher (design section 3).
    wstring installedExe, have;
    bool offerReplace = false, otherInstalled = false;
    LauncherStartDecision start;
    if (!uninstall && LauncherInstalled(&installedExe) && !gl.paths.installedCopy && !gl.paths.portable && !IEq(FullPath(installedExe), FullPath(SelfPath()))) {
        have = RegString(UninstallHive(), (UninstallRoot() + kLauncherEntry).c_str(), L"DisplayVersion");
        // rc5: the build decides at the same version, never the version string alone (wg_core DecideLauncherStart)
        start = DecideLauncherStart(SelfPath(), g.ctx.data->version, installedExe, have);
        offerReplace = start.choice == LauncherStart::OfferReplace;
        otherInstalled = start.choice != LauncherStart::RunHere;  // an unreadable installed copy: this copy runs (rc5)
        LogLine(g.ctx, L"start: " + start.why + L"; installed copy " + installedExe + L": " +
                           (offerReplace ? L"this copy offers to replace it" : otherInstalled ? L"opening the installed copy" : L"this copy runs"));
    }
    // One window at a time (two would write over each other's launcher.ini and could work in the same game): a second start
    // brings the open window forward, and from Apps & features asks the uninstall question there. Only a newer download
    // that offers to replace the installed launcher opens beside it, for that question only: whatever the answer, it hands
    // over to the installed copy and closes (HandOverToInstalled; the open one is asked to close once it is replaced).
    HANDLE one = CreateMutexW(nullptr, FALSE, L"Local\\WrongGPU.Launcher.Window");  // held until this program ends
    windowLock = &one;
    if (one && GetLastError() == ERROR_ALREADY_EXISTS && !offerReplace)
        if (HWND other = FindWindowW(L"WrongGPULauncher", nullptr)) {
            if (IsIconic(other)) ShowWindow(other, SW_RESTORE);
            SetForegroundWindow(other);
            if (uninstall) PostMessageW(other, WM_APP_UNINSTALL, 0, 0);
            return EXIT_OK;
        }
    // the same or an older release next to the installed launcher: the installed copy opens instead (a second copy that is
    // not installed would work on the same data, and an older one would put its older files into games)
    if (otherInstalled && !offerReplace) {
        if (one) { CloseHandle(one); one = nullptr; }  // the installed copy takes the one-window lock
        if ((INT_PTR)ShellExecuteW(nullptr, L"open", installedExe.c_str(), nullptr, Parent(installedExe).c_str(), SW_SHOWNORMAL) > 32) {
            LogLine(g.ctx, L"start: WrongGPU " + have + L" is installed (" + installedExe + L"); opened it instead of this copy (" + g.ctx.data->version + L"): " + start.why);
            return EXIT_OK;
        }
        // rc5: it didn't start: this copy runs (it used to say where the installed copy is and close)
        LogLine(g.ctx, L"start: the installed copy " + installedExe + L" didn't start; this copy runs");
        one = CreateMutexW(nullptr, FALSE, L"Local\\WrongGPU.Launcher.Window");  // the one-window lock back
    }
    if (gl.paths.installedCopy) RefreshLauncherEntry(g.ctx, SelfPath());  // its entry up to date (version, size, the uninstall command)
    // a newer download that only asks "Update WrongGPU to X?" reads and writes nothing of the installed copy's data meanwhile
    // (no conversion's model put back under an open window's conversion, no scan, health check or saved list and place)
    replaceAsking = offerReplace;
    CreateLauncherWindow(true);
    SetHealth(false, GpuInfo());
    if (!offerReplace) {
        RecoverModelKeep();  // a conversion that did not finish: the model it set aside back first (nothing else runs yet)
        TakeModelStateOnDisk();  // the model data as it was left, until StartHealth has checked it
    }
    v.executorLine = L"Checking the graphics card\u2026";
    v.urlText = gl.global.updateUrl;
    v.launcherInstalled = LauncherInstalled();
    // the first frame is the Library looking for games (its skeleton) or the first run's sheet, never "No games found yet"
    if (offerReplace) { v.page = P_LIBRARY; v.scanning = true; }  // the question's backdrop: the skeleton (no scan starts)
    else if (uninstall) {  // from Apps & features: Settings, with its uninstall question (posted once the window shows)
        v.page = P_SETTINGS;
        openedToUninstall = true;  // its Cancel closes the window again
        StartScan();
    } else if (gl.global.firstRunDone) {
        LoadLibrarySnapshot();  // the last session's list at once (SHIP-BAR 3), while the scan reads the games again
        StartScan();
    } else {
        v.first.installOffer = !LauncherInstalled() && !gl.paths.portable;
        v.first.choice = v.first.installOffer ? 1 : 0;  // a radio choice has its default: install (Continue says so)
        StartFirstRun();
    }
    // shown cloaked until its first frame is presented (the Direct3D device is made in that first paint): no empty white
    // or black window first
    const BOOL cloak = TRUE, uncloak = FALSE;
    const bool cloaked = SUCCEEDED(DwmSetWindowAttribute(v.hwnd, 13 /* DWMWA_CLOAK */, &cloak, sizeof cloak));
    ShowWindow(v.hwnd, startShow);  // where and how it was last time (or as its shortcut asks)
    UpdateWindow(v.hwnd);
    if (cloaked) DwmSetWindowAttribute(v.hwnd, 13 /* DWMWA_CLOAK */, &uncloak, sizeof uncloak);
    if (!offerReplace) StartHealth(true);
    if (offerReplace) {
        // ui_words.cpp; Not now (rc5): this copy runs for this session, as a launcher of its own (never the older build
        // silently): what the question's backdrop held back starts now
        if (AnswerReplaceOffer(one, installedExe, have, start) && IsWindow(v.hwnd)) {
            replaceAsking = false;
            RecoverModelKeep();
            TakeModelStateOnDisk();
            v.scanning = false;
            if (gl.global.firstRunDone) {
                LoadLibrarySnapshot();
                StartScan();
            } else {
                v.first.installOffer = false;  // the launcher is installed already (the copy this one didn't replace)
                v.first.choice = 0;
                StartFirstRun();
            }
            StartHealth(true);
        }
    }
    if (!IsWindow(v.hwnd)) {}  // handed over to the installed copy (or closed): nothing more starts here
    else if (uninstall) PostMessageW(v.hwnd, WM_APP_UNINSTALL, 0, 0);  // from Apps & features: Settings, with its uninstall question
    else if (gl.global.firstRunDone) StartUpdateCheck();  // the first run's end starts it otherwise (FinishFirstRun)
    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0)) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    windowLock = nullptr;
    if (bigIcon) DestroyIcon(bigIcon);
    if (smallIcon) DestroyIcon(smallIcon);
    // The window is gone. Workers still running only read (a scan, the health read, a release check: an operation or a
    // task keeps the window open), but they use g.ctx and the log, which the C runtime destroys once this returns: wait for
    // them a moment, and end the process at once if one is still stuck (a slow network, a disk that does not answer).
    // a HIP probe of the first run's checks or the health check (a child of this exe) ends now: none outlives this process,
    // writes the temporary log again or holds a downloaded exe open (its worker then ends at once)
    StopHipProbes();
    const ULONGLONG until = GetTickCount64() + 3000;
    while (workers > 0 && GetTickCount64() < until) Sleep(10);
    DropFirstRunLog();  // closed before the first run's choice: nothing of this start stays (its log was in the temp folder)
    // the moved exe of an uninstall (the window may have stayed open past the hidden cleanup's hour): tried again from now
    RetryCleanupsAtExit(g.ctx);
    if (workers > 0) {
        // the process ends without its destructors: the OptiScaler package an operation unpacked to %TEMP% goes now (~Context)
        if (!g.ctx.opti.tempDir.empty()) RemoveTree(g.ctx.opti.tempDir);
        TerminateProcess(GetCurrentProcess(), EXIT_OK);
    }
    return EXIT_OK;
}

// Fixture screenshots of the window (development): the Library, the Library filtered to "Needs attention", a game's
// page (plain, with its "more" menu, with unsaved changes), Settings and the first-run summary, at each scale.
}  // namespace wg::app
