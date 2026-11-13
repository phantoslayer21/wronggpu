// SPDX-License-Identifier: MIT
// WrongGPU launcher window: Borderlands 4 and Hogwarts Legacy's Engine.ini setting for Ray Reconstruction (Brief 16, owner
// 2026-11-04; wg_core/rrconfig.h). Their Ray Reconstruction needs one line in the game's own Engine.ini, and Borderlands 4
// removes that file on every start unless it is read-only.
// The question: once an install, repair or update is done and the window is free (no other question open, nothing running),
// "<game> needs one more setting for Ray Reconstruction": Fix it for me (rr_config_fix=on: an Apply writes it), Show me how
// (rr_config_fix=off: the game's page opens scrolled to its Engine.ini card, with the steps, the folder, the line and a file
// check, the keyboard on Open folder: ShowRrConfigSteps) or Not now (nothing changes; asked again after the next install, repair or update). A game whose rr_config_fix is
// on or off is never asked; the game page's "Engine.ini setting" row changes it at any time (UI review #2).
// The row's buttons (RrConfigAction): Open folder, Copy (the line), Check again (the file read again), Fix it again (WrongGPU
// manages the line and the file check says it is gone or not read-only: the saved choice applied again).
// Fix it for me before the game's first start (r2b #1): nothing is written (the settings folder isn't known yet); the card says
// "Will be added after you start the game once", and the choice is applied once the folder is there (ApplyRrConfigWhenDue: the
// launcher opening, the game's page shown, Check again; Repair/Update apply it too). Only where it needs no permission (r2c): a
// game in a protected folder (GameEntry::rrConfigProtected) would get a permission prompt nobody clicked for, so the window
// never starts it there; the card says "Ready to add" and Fix it now (the player's click) or Apply, Repair, Update add it.
// Written for this project.
#include "app/launcher_app.h"
#include "app/launcher_window.h"

#include <shellapi.h>
#include <shlobj.h>

#include <set>

#include "wg_core/rrconfig.h"

namespace wg::app {
using namespace wg::app::lui;
namespace win {

// launcher_ui.cpp
int RunDialog(int icon, const wstring& body, std::vector<DialogButton> buttons, int cancel);
void StartOp(OpRequest r);
GameEntry* EntryOf(const wstring& dir, const wstring& exe);
extern int modalDepth;

bool CopyToClipboard(const wstring& t) {
    if (!OpenClipboard(v.hwnd)) return false;
    EmptyClipboard();
    bool ok = false;
    if (HGLOBAL h = GlobalAlloc(GMEM_MOVEABLE, (t.size() + 1) * sizeof(wchar_t))) {
        if (void* p = GlobalLock(h)) {
            memcpy(p, t.c_str(), (t.size() + 1) * sizeof(wchar_t));
            GlobalUnlock(h);
            ok = SetClipboardData(CF_UNICODETEXT, h) != nullptr;
        }
        if (!ok) GlobalFree(h);
    }
    CloseClipboard();
    return ok;
}

// "Show me how": the game's page (opened unless it is the one shown), scrolled to its Engine.ini card with the keyboard on
// the first step's Open folder once the steps show there (rrcfgReveal: PaintGame, after the Apply that sets rr_config_fix=off).
// The card is under the checklist, the Upscaler and Ray Reconstruction: two screens down at the default size.
void ShowRrConfigSteps(const wstring& dir) {
    const bool here = v.page == P_GAME && IEq(v.cur.dir, dir);
    wstring name;
    for (size_t i = 0; i < v.games.size(); ++i)
        if (IEq(v.games[i].dir, dir)) {
            name = v.games[i].name;
            if (!here) OpenGame((int)i);
            break;
        }
    if (here) name = v.cur.name;
    v.rrcfgReveal = dir;
    // one line: the toast stays clear of the card's file check at the default size (the check row says what Check again does)
    ShowToast(L"Here are the steps for " + name + L"'s Engine.ini.", 0);
    Invalidate();
}

namespace {
struct Ask { wstring dir, exe; Profile saved; };
std::vector<Ask> asks;
const UINT_PTR kAskTimer = 0x5243;

void CALLBACK Tick(HWND hwnd, UINT, UINT_PTR id, DWORD);
void Answer(const Ask& a, int c);
void AskNow() {
    while (!asks.empty() && !v.busy) {
        const Ask a = asks.front();
        asks.erase(asks.begin());
        const RrConfigGame* rg = RrConfigGameFor(a.exe);
        if (!rg || !RrConfigAskable(a.dir, a.exe)) continue;  // (its RR files gone meanwhile: r7 #5)
        const int c = RunDialog(0, RrConfigQuestionText(a.dir, a.exe), {{L"Not now", 0}, {L"Show me how", 2}, {L"Fix it for me", 1, true}}, 0);
        if (c == 0 || !v.live) continue;  // Not now: nothing changes (asked again after the next install, repair or update)
        if (v.busy) { asks.insert(asks.begin(), a); break; }  // something started meanwhile: asked again once it is done
        Answer(a, c);
    }
    if (!asks.empty() && v.hwnd) SetTimer(v.hwnd, kAskTimer, 400, Tick);
}
// The question's answer (Fix it for me 1, Show me how 2): the choice saved alone, rr_config_fix into profile.ini as it is when
// the operation runs, then Engine.ini as it says (OpRequest::rrConfigSet), never a full Apply of the profile captured when the
// question was asked (r4 #4: a settings change made since would be put back). Show me how shows the steps first (r3 #6)
void Answer(const Ask& a, int c) {
    {
        OpRequest r; r.op = Op::Apply; r.dir = a.dir; r.exe = a.exe; r.rrConfigOnly = true; r.rrConfigSet = c == 1 ? "on" : "off";
        if (c == 2) {
            // Show me how: the game's page at its steps, with the file check. The steps show at once, as Myself chosen on the page
            // (unsaved until saved: rrcfgMyselfDir), never only after the save (r3 #6). Saving it writes the game's folder: where
            // that needs administrator permission nothing is run (no prompt for reading steps; the page's Apply saves it, Windows
            // asks then), elsewhere it is saved now, and a folder protected since runs nothing either (noPrompt)
            v.rrcfgMyselfDir = a.dir;
            const GameEntry* e = EntryOf(a.dir, a.exe);
            const bool prot = (e && e->rrConfigProtected) || (IEq(v.cur.dir, a.dir) && v.cur.rrConfigProtected);
            ShowRrConfigSteps(a.dir);
            // its page was already shown (no page read to come: ShowEntryWith puts Myself there once its files are read)
            if (!v.rrcfgMyselfDir.empty() && v.page == P_GAME && IEq(v.cur.dir, a.dir) && !v.pageReading) {
                if (v.saved.get("rr_config_fix") == "ask") SetSetting(v.edit, "rr_config_fix", "off", nullptr);
                v.rrcfgMyselfDir.clear();
            }
            if (prot) {
                if (!v.fixture) try { LogLine(g.ctx, L"rr_config_fix: Show me how for " + a.dir + L" (protected folder): the steps shown, the choice not saved yet (no permission prompt)"); } catch (...) {}
                return;
            }
            r.noPrompt = true;
        }
        StartOp(r);
    }
}
// the timer's own callback (not the window's WM_TIMER): the question once the window is free
void CALLBACK Tick(HWND hwnd, UINT, UINT_PTR id, DWORD) {
    if (!v.live || !v.hwnd) { KillTimer(hwnd, id); asks.clear(); return; }
    if (v.busy || firstRunBusy || modalDepth > 0 || v.dialog.open) return;  // later
    KillTimer(hwnd, id);
    AskNow();
}
}  // namespace

// ---- the choice made before the game's first start, applied once its settings folder is there (r2b #1) ----------------
namespace {
std::vector<std::pair<wstring, wstring>> due;  // (dir, exe), in the order found
std::set<wstring> dueTried;                    // lower-case dirs: once a session (an Apply that started and can't add it isn't repeated)
const UINT_PTR kDueTimer = 0x5244;
// A protected game's line waits for the player's click ("Ready to add"): said once a session, when the window's own apply
// skips it (the launcher opening), unless its page is shown (its card says it). One toast for all of them (r3 #5)
bool readySaid = false;
void SayReadyToAdd(const wstring& dir) {
    if (readySaid || (v.page == P_GAME && IEq(v.cur.dir, dir))) return;
    const wstring t = ReadyToAddToast();
    if (t.empty()) return;
    readySaid = true;
    ShowToast(t, 0);
}
// the first game due: its saved choice applied (its own page: as Fix it again, unsaved changes there kept). A running game: left
// for its entry read again once it closes. True: an Apply started
bool StartDue() {
    while (!due.empty() && !v.busy) {
        // the last session's list is shown (its entries not read yet): later, the timer tries again (nothing marked as tried)
        if (v.listCached) return false;
        const auto d = due.front();
        due.erase(due.begin());
        const GameEntry* e = EntryOf(d.first, d.second);
        if (!e || !e->hasRecord || e->status == GameStatus::Running || dueTried.count(Lower(d.first))) continue;
        // without WrongGPU's RR files the line serves nothing of WrongGPU's: never added by the window itself (r7 #5; the card says so,
        // and the player's Fix it now, Apply, Repair or Update still adds it)
        if (!e->rrInstalled) continue;
        // its folder needs administrator permission: never a prompt the player didn't click for; the card says "Ready to add"
        // (r2c). The worker checks it once more (OpRequest::noPrompt), so a folder that turned protected since is never asked either
        if (e->rrConfigProtected || (IEq(v.cur.dir, d.first) && v.cur.rrConfigProtected)) { SayReadyToAdd(d.first); continue; }
        try { LogLine(g.ctx, L"rr_config_fix: the settings folder of " + e->name + L" is there now: the choice made before is applied"); } catch (...) {}
        // Engine.ini alone, as the saved profile has the choice (r3 #1): no other settings file is written, so a value changed
        // outside the launcher (its own Keep / Use buttons decide) and the page's unsaved changes stay as they are
        OpRequest r; r.op = Op::Apply; r.dir = d.first; r.exe = d.second; r.noPrompt = true; r.rrConfigOnly = true;
        StartOp(r);
        // tried once a session only when it really started (r3 #7): an apply that couldn't start is asked again later
        if (v.busy) dueTried.insert(Lower(d.first));
        else due.insert(due.begin(), d);
        return v.busy;
    }
    return false;
}
void CALLBACK DueTick(HWND hwnd, UINT, UINT_PTR id, DWORD) {
    if (!v.live || !v.hwnd) { KillTimer(hwnd, id); due.clear(); return; }
    if (v.busy || firstRunBusy || modalDepth > 0 || v.dialog.open || !asks.empty()) return;  // later
    StartDue();
    if (due.empty()) KillTimer(hwnd, id);
}
}  // namespace

wstring ReadyToAddToast() {
    std::vector<const GameEntry*> ready;
    for (auto& g2 : v.games) if (RrConfigReadyToAdd(g2)) ready.push_back(&g2);
    if (ready.empty()) return L"";
    return ready.size() == 1 ? ready[0]->name + L"'s Ray Reconstruction setting is ready to add. Click Fix it now: Windows asks for permission first."
                             : std::to_wstring(ready.size()) + L" games have their Ray Reconstruction setting ready to add. Click Fix it now on each: "
                                                               L"Windows asks for permission first.";
}

void ApplyRrConfigWhenDue(const wstring& dir, const wstring& exe) {
    if (!v.live || !v.hwnd || dueTried.count(Lower(dir))) return;  // (the ui-dump: ApplyRrConfigDueNow)
    for (auto& d : due) if (IEq(d.first, dir)) return;
    due.push_back({dir, exe});
    SetTimer(v.hwnd, kDueTimer, 400, DueTick);
}
bool ApplyRrConfigDueNow(const wstring& dir, const wstring& exe) {
    due.insert(due.begin(), {dir, exe});
    return StartDue();
}
void RrConfigNeedsClick(const wstring& dir) {
    dueTried.insert(Lower(dir));  // not asked again this session; the player's click does it
    if (GameEntry* e = EntryOf(dir, L"")) e->rrConfigProtected = true;
    if (IEq(v.cur.dir, dir)) v.cur.rrConfigProtected = true;
}
void ForgetRrConfigDue(const wstring& dir) { dueTried.erase(Lower(dir)); }

void AnswerRrConfigQuestion(const wstring& dir, const wstring& exe, const Profile& saved, int choice) { Answer(Ask{dir, exe, saved}, choice); }

// The question's text: in a protected folder it says Fix it for me brings Windows' permission prompt (r4 #9)
wstring RrConfigQuestionText(const wstring& dir, const wstring& exe) {
    const RrConfigGame* rg = RrConfigGameFor(exe);
    if (!rg) return L"";
    const GameEntry* e = EntryOf(dir, exe);
    const bool prot = (e && e->rrConfigProtected) || (IEq(v.cur.dir, dir) && v.cur.rrConfigProtected);
    // before the game's first start (its settings folder not there: the page's file check -1) the prompt comes later, at Fix it
    // now (r5 #9): the page's check when it is this game's, else the folder looked up (never under ui-render: its state only)
    bool first = false;
    if (prot) {
        if (IEq(v.cur.dir, dir) && v.rrcfg != -2) first = v.rrcfg == -1;
        else if (!v.fixture) { bool known = true; try { RrConfigFile(g.ctx, *rg, nullptr, &known); } catch (...) {} first = !known; }
    }
    return RrConfigNotice(*rg, prot, first);
}

// Whether the question may come for this game (r7 #5): Borderlands 4 / Hogwarts Legacy with WrongGPU's RR files (an install without
// them, offline or not the WrongGPU edition, has no RR of WrongGPU's for the line to serve: no question, no permission prompt for it)
bool RrConfigAskable(const wstring& dir, const wstring& exe) {
    if (!RrConfigGameFor(exe)) return false;
    // (rc6: never where WrongGPU's RR isn't offered: the updated Hogwarts Legacy)
    if (const GameEntry* e = EntryOf(dir, exe)) return e->rrInstalled && !e->rrNotOffered;
    return IEq(v.cur.dir, dir) && v.cur.rrInstalled && !v.cur.rrNotOffered;
}

// After an operation (WM_APP_OPDONE): the game's profile as read again on the worker (its entry read again before this).
void AskRrConfigLater(const wstring& dir, const wstring& exe, const Profile& saved) {
    if (!RrConfigAskable(dir, exe) || saved.get("rr_config_fix") != "ask" || !v.hwnd) return;
    for (auto& a : asks) if (IEq(a.dir, dir)) return;
    asks.push_back({dir, exe, saved});
    SetTimer(v.hwnd, kAskTimer, 400, Tick);
}

// The game page's "Engine.ini setting" steps: Open folder (never made here: the game makes it at its first start), Copy (the
// line on the clipboard), Check again (the file read again; a small file in the player's profile, read on a click).
void RrConfigAction(int id) {
    const RrConfigGame* rg = RrConfigGameFor(v.cur.exe);
    if (!rg) return;
    wstring file = v.rrcfgFile;
    if (file.empty()) { try { file = RrConfigFile(g.ctx, *rg, nullptr); } catch (...) {} }
    if (id == ID_G_RRCFG_OPEN) {
        // never made here: before the game's first start the sub-folder is a guess (Hogwarts Legacy uses WindowsNoEditor, the
        // guess is Windows), and a file made in the wrong one would be edited for nothing (UI review r2 #7)
        const wstring dir = file.empty() ? wstring() : Parent(file);
        if (dir.empty() || !IsDir(dir)) {
            v.rrcfg = -1;  // the file check says so too
            ShowToast(L"The game's settings folder isn't there yet. Start the game once, then come back.", 0);
            Invalidate();
            return;
        }
        if ((INT_PTR)ShellExecuteW(nullptr, L"open", dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL) <= 32)
            ShowToast(L"The folder couldn't be opened.", 2);
        return;
    }
    if (id == ID_G_RRCFG_COPY) {
        ShowToast(CopyToClipboard(RrConfigLineW()) ? L"The line is on the clipboard. Paste it under [SystemSettings] in Engine.ini."
                                                   : L"The line couldn't be put on the clipboard. Type it as the page shows it.", 0);
        return;
    }
    // WrongGPU's line gone or the file no longer read-only: the saved choice applied again (r2 #6); or "Ready to add" (Fix it
    // now: the game's folder is protected, so the permission prompt follows this click, r2c)
    // Engine.ini alone (r3 #1): the page's other values, saved or not, stay as they are
    if (id == ID_G_RRCFG_FIX) {
        if (v.saved.get("rr_config_fix") != "on" || lui::PageWritesWait()) return;  // (waits for an operation or the game, as Apply)
        OpRequest r; r.op = Op::Apply; r.dir = v.cur.dir; r.exe = v.cur.exe; r.rrConfigOnly = true;
        StartOp(r);
        return;
    }
    if (id == ID_G_RRCFG_CHECK) {
        int state = -1;
        wstring why;
        try { state = RrConfigLineState(g.ctx, *rg, &file, &why); } catch (...) {}
        v.rrcfg = state;
        v.rrcfgFile = file;
        v.rrcfgWhy = why;
        // WrongGPU's choice made before the game's first start (r2b #1): the folder there now applies it; not yet, it waits
        const bool waiting = v.saved.get("rr_config_fix") == "on" && !v.rrcfgFixed;
        // ...in a protected folder, or without WrongGPU's RR files (r7 #5): added on the player's click, never by itself (r2c)
        const bool click = waiting && (v.cur.rrConfigProtected || !v.cur.rrInstalled);
        ShowToast(state == 1 ? L"Engine.ini has the line. Ray Reconstruction can run in " + wstring(rg->name) + L"."
                  : state == 3 ? (v.saved.get("rr_config_fix") == "off"  // Myself: the player's own file, which the game may read fine (r4 #6)
                                      ? L"WrongGPU's check couldn't read Engine.ini: " + why + L". Check it by hand in Notepad."
                                      : L"Engine.ini couldn't be read: " + why + L".")  // r3 #8
                  : state == 2 ? L"Engine.ini has the line, but it isn't read-only yet."
                  : state == 0 ? (click ? L"The game's settings folder is there now. Click Fix it now to add the line."
                                 : waiting ? L"The game's settings folder is there now. WrongGPU adds the line." : L"Engine.ini doesn't have the line yet.")
                  : waiting && !file.empty() ? (click ? L"The game's settings folder isn't there yet. Start the game once, then click Fix it now here to add the line."
                                                      : L"The game's settings folder isn't there yet. WrongGPU adds the line after you start the game once.")
                               : L"The game's settings folder isn't there yet. Start the game once, then check again.",
                  state == 1 ? 1 : 0);
        if (waiting && !click && state >= 0) ApplyRrConfigWhenDue(v.cur.dir, v.cur.exe);
        Invalidate();
    }
}

}  // namespace win
}  // namespace wg::app
