// SPDX-License-Identifier: MIT
// WrongGPU launcher window: what a click, Enter or Space on a widget does (ActivateWidget; Activate in launcher_ui.cpp calls
// it): the Library, a game's page, Settings, the first-run summary, menus and dialogs. Split from launcher_ui.cpp (the spec
// side's 256 KB scan limit) with no change in behaviour. Written for this project.
#include "app/launcher_ui_state.h"

#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>

#include "wg_core/gamerules.h"
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
// launcher_ui.cpp
void StartOp(OpRequest r);
}  // namespace win
using namespace win;

namespace lui {
void ActivateWidget(int id) {
    if (id >= ID_DIALOG && id < ID_DIALOG + 100) {
        const size_t i = (size_t)(id - ID_DIALOG);
        if (i < v.dialog.buttons.size()) { v.dialog.result = v.dialog.buttons[i].result; v.dialog.open = false; }
        Invalidate();
        return;
    }
    if (id >= ID_CAP_MIN && id <= ID_CAP_CLOSE) {  // the caption buttons (the mouse: WM_NCLBUTTONUP; UI Automation: Invoke)
        if (!v.live || !v.hwnd) return;
        const WPARAM cmd = id == ID_CAP_MIN ? SC_MINIMIZE : id == ID_CAP_CLOSE ? SC_CLOSE : IsZoomed(v.hwnd) ? SC_RESTORE : SC_MAXIMIZE;
        PostMessageW(v.hwnd, WM_SYSCOMMAND, cmd, 0);
        return;
    }
    if (id >= ID_MENU && id < ID_MENU + 100) {
        const size_t i = (size_t)(id - ID_MENU);
        const Menu m = v.menu;
        v.menu.open = false;
        clickGuardAt = GetTickCount64();  // the second click of a double click on an item does not press the control under it
        v.focus = m.owner;  // chosen with the mouse too: the next Tab goes on from the menu's control
        Invalidate();
        if (i < m.items.size() && m.items[i].enabled && !m.items[i].sep) MenuCommand(m.owner, m.items[i].id);
        return;
    }
    if (id >= ID_TILE && id < ID_TILE + kTileSpan) { OpenGame(id - ID_TILE); return; }
    if (id >= ID_S_THEME0 && id < ID_S_THEME0 + 3) {
        static const wchar_t* const t[] = {L"system", L"light", L"dark"};
        gl.global.appearance = t[id - ID_S_THEME0];
        SaveSettings();
        ApplyTheme();
        return;
    }
    if (id >= ID_S_ACCENT0 && id < ID_S_ACCENT0 + A_COUNT) {  // the accent colour: saved, and the window follows at once
        gl.global.accent = AccentKey(id - ID_S_ACCENT0);
        SaveSettings();
        ApplyTheme();
        return;
    }
    if (id == ID_FEAT0 + 1 || id == ID_FEAT0 + 2) {  // the featured banner: the tile's action (Play, Repair, Update), the game's page
        if (v.featIndex < 0 || v.featIndex >= (int)v.games.size()) return;
        if (id == ID_FEAT0 + 2) OpenGame(v.featIndex);
        else Activate(ID_TILEQA + v.featIndex);
        return;
    }
    if (id >= ID_S_INDEX0 && id < ID_S_INDEX0 + 3) {  // Settings' section index: the section to the top, and the keyboard with it
        const float to = SettingsSectionScroll(id - ID_S_INDEX0);
        ScrollBy(to - v.target[P_SETTINGS]);
        // as in-page navigation has it: the section's first control takes the focus, so the next Tab goes on from there (it
        // did not scroll back to the page's top). Its heading sits at the scroll + the toolbar + 18 (SettingsSectionScroll).
        const float headY = to + kBarH + 18 - 4;
        const Hit* first = nullptr;
        for (const Hit* h : Focusables()) {  // Tab's stops: a segmented control's is its chosen segment
            if (ToolbarHit(h->id) || (h->id >= ID_S_INDEX0 && h->id < ID_S_INDEX0 + 3)) continue;
            if (h->full.y + h->scroll >= headY) { first = h; break; }
        }
        if (first) FocusId(first->id);
        return;
    }
    if (id >= ID_S_FOLDERRM0 && id < ID_S_FOLDERRM0 + kMaxFolderRows) {  // an added game folder's Remove
        const size_t i = (size_t)(id - ID_S_FOLDERRM0);
        if (i >= gl.global.extraFolders.size()) return;
        const wstring f = gl.global.extraFolders[i];
        if (!Confirm(L"Remove this game folder from the list?\n\n" + PlaceWords(f) +
                     L"\n\nGames found only in this folder leave the Library; games with WrongGPU stay. Nothing on disk changes; add it again any time.", 0, L"Remove", L"Cancel", true))
            return;
        auto& list = gl.global.extraFolders;
        // this row only (a list saved with a folder twice keeps its other row); the list changed meanwhile: the first same one
        auto at = i < list.size() && list[i] == f ? list.begin() + (long long)i : std::find_if(list.begin(), list.end(), [&](const wstring& x) { return IEq(x, f); });
        if (at != list.end()) list.erase(at);
        SaveSettings();
        StartScan();
        Invalidate();
        return;
    }
    if (id >= ID_TILEQA && id < ID_TILEQA + kTileSpan) {
        const int i = id - ID_TILEQA;
        if (i >= (int)v.games.size()) return;
        const GameEntry e = v.games[i];
        // a running game's pill waits until it closes (its Play or Install, off): a second press read against the frame
        // before Play (a double click, Enter twice, two Invokes) does nothing, never opens its page
        if (e.status == GameStatus::Running) return;
        if (e.adoptable && (e.status == GameStatus::Installed || e.status == GameStatus::TurnedOff)) { OnAdopt(e); return; }
        // Fix it now (r3 #5): the Engine.ini line alone, as the game's page's Fix it now does; the permission prompt follows this click
        if (RrConfigReadyToAdd(e)) {
            if (v.busy || v.listCached) return;  // (the pill is off then, BusyWhy says why)
            OpRequest r; r.op = Op::Apply; r.dir = e.dir; r.exe = e.exe; r.rrConfigOnly = true;
            StartOp(r);
            return;
        }
        switch (e.status) {
            case GameStatus::Installed: case GameStatus::TurnedOff:
                if (LockupPlays(e)) Play(e);  // what its pill says (a gone .exe: Play's note says so)
                else OpenGame(i);
                return;
            case GameStatus::UpdateAvailable: OnInstall(e, Op::Update); return;
            case GameStatus::NeedsRepair: OnInstall(e, Op::Repair); return;
            case GameStatus::Eligible: case GameStatus::PlayersOptiScaler: case GameStatus::OldInstall: OnInstall(e, Op::Install); return;
            default: OpenGame(i); return;
        }
    }
    if (id >= ID_G_EXEC0 && id < ID_G_EXEC0 + 3) { static const char* const x[] = {"auto", "dx12", "hip"}; SetSetting(v.edit, "executor", x[id - ID_G_EXEC0], nullptr); Invalidate(); return; }
    if (id >= ID_G_PRE0 && id < ID_G_PRE0 + 2) { static const char* const x[] = {"scale", "original"}; SetSetting(v.edit, "pre_exposure", x[id - ID_G_PRE0], nullptr); Invalidate(); return; }
    if (id >= ID_G_RRMODE0 && id < ID_G_RRMODE0 + 3) { static const char* const x[] = {"exact", "accurate", "fast"}; SetSetting(v.edit, "rr_mode", x[id - ID_G_RRMODE0], nullptr); Invalidate(); return; }
    if (id >= ID_G_FPS0 && id < ID_G_FPS0 + 3) { static const char* const x[] = {"keep", "on", "off"}; SetSetting(v.edit, "fps_overlay", x[id - ID_G_FPS0], nullptr); Invalidate(); return; }
    if (id >= ID_G_OPTI0 && id < ID_G_OPTI_END) {
        const size_t row = (size_t)(id - ID_G_OPTI0) / 10;
        const int opt = (id - ID_G_OPTI0) % 10;
        if (row >= v.opti.editable.size()) return;
        const OptiKind kind = v.opti.editable[row].def ? v.opti.editable[row].def->kind : OptiKind::Number;
        if (kind == OptiKind::Bool) { static const char* const b[] = {"auto", "true", "false"}; if (opt < 3) v.optiValues[row] = b[opt]; Invalidate(); return; }
        if (kind == OptiKind::Key) {
                std::vector<MenuItem> items;
            items.push_back({L"Auto (Insert)", M_OPTI0, true, false, false, v.optiValues[row] == "auto" || v.optiValues[row].empty()});
            for (int k = 0; k < 20; ++k) {
                char b[16]; snprintf(b, sizeof b, "0x%02X", kMenuKeys[k]);
                items.push_back({KeyName(b), M_OPTI0 + 1 + k, true, false, false, _stricmp(v.optiValues[row].c_str(), b) == 0});
            }
            for (auto& h : v.hits) if (h.id == id) OpenMenu(id, h.r, items, true, 200);
            return;
        }
        // a number: its EDIT has the keyboard (not while its row is locked: the game runs, an operation runs)
        if (row < v.edOpti.size() && !v.edOpti[row].enabled) return;
        FieldNow();
        if (row < v.edOpti.size() && v.edOpti[row].h) SetFocus(v.edOpti[row].h);
        return;
    }
    const GameEntry e = v.cur;
    switch (id) {
        case ID_NAV_LIBRARY: Navigate(P_LIBRARY); return;
        case ID_NAV_SETTINGS:
            if (v.page == P_SETTINGS) return;  // there already: an address being typed stays
            // an address typed but not saved stays too when Settings opens again from another page (Save or Check now
            // saves it; only switching the release check off and on puts the saved address back in the field)
            if (!Navigate(P_SETTINGS)) return;
            if (!v.fixture) v.launcherInstalled = LauncherInstalled();
            return;
        case ID_RESCAN:  // it keeps the focus while it looks ("Looking for games…": a press then does nothing); the outcome is said
            if (v.scanning) return;
            rescanAsked = true;
            ForgetRrDetection();  // a game updated meanwhile may ship Ray Reconstruction now (or no longer)
            UiaNotify(L"Looking for games\u2026");
            StartScan();
            return;
        case ID_ADDGAME: {
            wstring p;
            if (!Browse(0, p)) return;
            // the game the chosen file leads to (rc6): a launcher stub's game, a root starter's game of the same name in
            // <Project>\Binaries\Win64 (Hogwarts Legacy) or bin\x64, an Unreal starter's shipping exe (ResolveGameExe)
            if (const wstring game = ResolveGameExe(p); !game.empty()) p = game;
            // a game listed already (by a store, or added before) is not added twice: said by its name, as Add a folder does
            for (auto& x : v.games)
                if (IEq(FullPath(x.exe), FullPath(p))) { ShowToast(x.name + L" is in the Library already.", 0); return; }
            for (auto& a : gl.global.addedGames)
                if (IEq(FullPath(a), FullPath(p))) { addedExe = p; StartScan(); return; }  // listed in launcher.ini, not shown yet: the scan says
            gl.global.addedGames.push_back(p);
            SaveSettings();
            addedExe = p;  // said once the scan lists it (WM_APP_SCAN)
            StartScan();
            return;
        }
        case ID_UPDATEALL: UpdateAll(); return;
        case ID_ATTNMORE:
            // from the keyboard: opened, the first lockup it revealed takes the focus (PaintAll), not the button, which moved
            // below them all; folded, the button (now under the few) is brought back into view
            attnShownBefore.clear();
            attnReveal = false;
            if (v.focus == ID_ATTNMORE && v.focusVisible) {
                if (!v.attnAll) {
                    attnReveal = true;
                    for (auto& h : v.hits) if (h.id >= ID_TILE && h.id < ID_TILE + kTileSpan) attnShownBefore.push_back(h.id);
                } else revealFocus = true;
            }
            v.attnAll = !v.attnAll;
            Invalidate();
            return;
        case ID_E_ADDGAME: Activate(ID_ADDGAME); return;
        case ID_E_SETTINGS: Activate(ID_S_ADDFOLDER); return;  // "Add a folder": the folder picker, the scan, the toast that says what it found
        case ID_E_CLEAR:  // the search field takes the focus for the next search (the button goes with the empty state)
            v.search.clear();
            if (v.edSearch.h) SetWindowTextW(v.edSearch.h, L"");
            FocusId(ID_SEARCH);
            Invalidate();
            return;
        case ID_SEARCH: FieldNow(); if (v.edSearch.h) SetFocus(v.edSearch.h); return;
        case ID_SEARCHCLEAR:  // the search field keeps the focus for the next search
            v.search.clear();
            if (v.edSearch.h) { SetWindowTextW(v.edSearch.h, L""); SetFocus(v.edSearch.h); }
            Invalidate();
            return;
        case ID_G_BACK: Navigate(P_LIBRARY); return;
        case ID_G_PRIMARY:
            switch (e.status) {
                case GameStatus::Eligible: case GameStatus::PlayersOptiScaler: case GameStatus::OldInstall: OnInstall(e, Op::Install); return;
                case GameStatus::UpdateAvailable: OnInstall(e, Op::Update); return;
                case GameStatus::NeedsRepair: OnInstall(e, Op::Repair); return;
                case GameStatus::Orphaned: OnInstall(e, Op::Uninstall); return;
                case GameStatus::Leftovers: OnLeftovers(e, true); return;
                case GameStatus::Installed: case GameStatus::TurnedOff: case GameStatus::Running: if (e.adoptable) OnAdopt(e); return;  // Adopt
                default: return;
            }
        case ID_G_SECOND: if (e.status == GameStatus::Leftovers) OnLeftovers(e, false); return;
        case ID_G_LEARN:  // the public page of the tool the game needs (REFramework), in the browser; nothing is downloaded
            if (!e.needsPage.empty() && !v.fixture) ShellExecuteW(v.hwnd, L"open", e.needsPage.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
            clickGuardAt = GetTickCount64();  // one browser tab: a double click's second click does not open another
            return;
        case ID_G_RECHECK: {  // this game's folder read again for the tool it needs (on a worker); the outcome is a toast
            if (rechecking || v.fixture || (v.busy && IEq(v.busyDir, e.dir))) return;
            rechecking = true;
            HWND hwnd = v.hwnd;
            const Profile defaults = gl.global.defaults;
            const wstring dir = e.dir, exe = e.exe, name = e.name, source = e.source;
            UiaNotify(L"Looking for " + e.needs + L"\u2026");
            Worker([hwnd, dir, exe, name, source, defaults] {
                auto* m = new EntriesMsg;
                m->recheck = true;
                m->name = name;
                try {
                    std::shared_lock<std::shared_mutex> opti(optiLock);
                    EntriesMsg::Item it{dir, exe, EntryFor(g.ctx, exe, name, source), PageData()};
                    ReadPageDataSafe(it.fresh, defaults, it.page);
                    m->items.push_back(std::move(it));
                } catch (...) {}
                if (!PostMessageW(hwnd, WM_APP_ENTRIES, 0, (LPARAM)m)) delete m;
            });
            return;
        }
        case ID_G_PLAY: {
            // unsaved changes: the game would start with its old settings; Apply first, or play without them (as leaving the
            // page asks)
            // While another game's operation or Collect logs runs, Apply waits: the question still comes (Cancel or Play
            // without them), never the game started with its old settings unasked
            if (Dirty() && v.live) {
                const bool wait = PageWritesWait();  // Apply waits: Cancel or Play without them
                const int a = RunDialog(0, PlayUnsavedText(e.name, wait), PlayUnsavedButtons(wait), 0);
                if (a == 0) return;
                if (a == 1) {
                    OnApply();
                    if (v.busy && IEq(v.busyDir, e.dir)) playAfterApply = e.exe;  // started once it is written (WM_APP_OPDONE)
                    return;
                }
            }
            Play(e);
            return;
        }
        case ID_G_MORE: {
            const std::vector<MenuItem> items = PageMoreMenu();
            // under the header's More from its left edge (option E); right-aligned from the compact header at the window's right
            for (auto& h : v.hits) if (h.id == ID_G_MORE) OpenMenu(ID_G_MORE, h.r, items, v.pinH > 0 || h.r.x + 280 > v.W - kGutter, 280);  // room for "Reset Upscaler settings to defaults"
            return;
        }
        case ID_G_MASTER: SetSetting(v.edit, "wronggpu", v.edit.get("wronggpu") == "off" ? "on" : "off", nullptr); Invalidate(); return;
        case ID_G_RRON:
            // rc5: a game whose RR runs only through its experimental opt-in (Spider-Man 2): the switch turns the opt-in off
            if (RrExperimentalHere()) { if (const GameRule* r = GameRuleFor(v.cur.exe)) SetSetting(v.edit, r->rrExperimentalSetting, "off", nullptr); Invalidate(); return; }
            SetSetting(v.edit, "rr_enabled", v.edit.get("rr_enabled") == "off" ? "on" : "off", nullptr); Invalidate(); return;
        case ID_G_RRREPAIR:  // an install from before this build: Set up (a repair) adds the RR files. The page's copy (e), as every
            // other caller: PrepareOp's inspection runs a message loop in which another game's page can replace v.cur
            if (!v.busy) {
                rrSetupNext = true;  // its verb: "Setting up..." (StartOp)
                OnInstall(e, Op::Repair);
                rrSetupNext = false;
                if (v.busy) SayResultAs(RES_RRREPAIR, e.dir);  // its toast says what it added
            }
            return;
        case ID_G_RRCONVERT: ConvertRr(L""); return;
        case ID_G_RRPREP: if (!v.busy) PrepareRrNow(e.dir, e.exe); return;
        case ID_G_SPARKLE: {
            const bool on = v.edit.get("sparkle_guard") != "off";
            SetSetting(v.edit, "sparkle_guard", on ? string("off") : (v.saved.get("sparkle_guard") != "off" ? v.saved.get("sparkle_guard") : string("on")), nullptr);
            Invalidate();
            return;
        }
        case ID_G_FILTER: {  // each choice with what it does under its name (FilterMenu)
            const std::vector<MenuItem> items = FilterMenu();
            for (auto& h : v.hits) if (h.id == ID_G_FILTER) OpenMenu(ID_G_FILTER, h.r, items, true, std::max(330.f, h.r.w));
            return;
        }
        case ID_G_DETAILS: v.detailsOpen = !v.detailsOpen; Invalidate(); return;
        case ID_G_LOCKED: v.lockedOpen = !v.lockedOpen; Invalidate(); return;
        case ID_G_OPTIMORE: v.optiMore = !v.optiMore; Invalidate(); return;
        case ID_G_ADVMORE: v.advOpen = !v.advOpen; Invalidate(); return;
        // Borderlands 4 / Hogwarts Legacy's Engine.ini setting (UI review #2): WrongGPU adds it, or the player does (an unsaved
        // change: Apply writes or takes out the line); the steps' buttons
        case ID_G_RRCFG0: case ID_G_RRCFG0 + 1:
            SetSetting(v.edit, "rr_config_fix", id == ID_G_RRCFG0 ? "on" : "off", nullptr); Invalidate(); return;
        case ID_G_RRCFG_OPEN: case ID_G_RRCFG_COPY: case ID_G_RRCFG_CHECK: case ID_G_RRCFG_FIX: RrConfigAction(id); return;
        // "Turn it on in the game": OptiScaler's FPS overlay on (an unsaved change: the capsule's Apply writes it)
        case ID_G_GUIDEFPS: SetSetting(v.edit, "fps_overlay", "on", nullptr); Invalidate(); return;
        case ID_G_LOG: MenuCommand(0, M_OPEN_LOG); return;
        case ID_G_COLLECT: MenuCommand(0, M_COLLECT); return;
        case ID_G_FOLDER: MenuCommand(0, M_OPEN_FOLDER); return;
        case ID_G_KEEP: {  // the file's values become the profile's (only those: other unsaved changes stay unsaved)
            if (PageWritesWait()) return;
            Profile p = v.saved;
            for (auto& x : v.outside) SetSetting(p, x.key, x.fileValue, nullptr);
            WriteProfileOnly(p);
            if (v.busy) SayResultAs(RES_KEEP, v.cur.dir);
            return;
        }
        case ID_G_REVERT: if (PageWritesWait()) return; WriteProfileOnly(v.saved); if (v.busy) SayResultAs(RES_REVERT, v.cur.dir); return;  // the profile's values go back into the files (only those)
        case ID_G_OKEEP: OnOptiFix(true); return;
        case ID_G_OREVERT: OnOptiFix(false); return;
        case ID_G_DISCARD: v.edit = v.saved; v.optiValues.clear(); for (auto& r : v.opti.editable) v.optiValues.push_back(r.value);
            for (size_t i = 0; i < v.edOpti.size() && i < v.optiValues.size(); ++i) if (v.edOpti[i].h) SetWindowTextW(v.edOpti[i].h, FromUtf8(v.optiValues[i]).c_str());
            UiaNotify(L"Changes discarded.", false, 1);  // the capsule goes away: said, not only seen
            clickGuardAt = GetTickCount64();  // ... at once: a double click's second click does not press the row under it
            Invalidate(); return;
        case ID_G_APPLY: OnApply(); return;
        case ID_G_NOTICE: v.notice.clear(); Invalidate(); return;
        // settings
        case ID_S_OPTIFILE: case ID_S_OPTIDIR: {
            wstring p;
            if (Browse(id == ID_S_OPTIFILE ? 3 : 4, p)) { gl.global.optiPackage = p; SaveSettings(); Invalidate(); }
            return;
        }
        case ID_S_MODELMORE: {  // a ready model's Options...: the same two ways as a model that is not ready, as a menu
            std::vector<MenuItem> items{{L"Choose a DLSS file\u2026", ID_S_MODELFILE, !v.busy, false, false, false, ic::cube},
                                        {L"Convert again", ID_S_RECONVERT, !v.busy, false, false, false, ic::sync}};
            for (auto& h : v.hits) if (h.id == ID_S_MODELMORE) OpenMenu(ID_S_MODELMORE, h.r, items, true, 230);
            return;
        }
        case ID_S_RRMORE: {  // ready Ray Reconstruction data's Options...: the same two ways as data that is not ready
            std::vector<MenuItem> items{{L"Choose a DLSS file\u2026", ID_S_RRFILE, !v.busy, false, false, false, ic::cube},
                                        {L"Convert again", ID_S_RRCONVERT, !v.busy, false, false, false, ic::sync}};
            for (auto& h : v.hits) if (h.id == ID_S_RRMORE) OpenMenu(ID_S_RRMORE, h.r, items, true, 230);
            return;
        }
        case ID_S_RRCONVERT: ConvertRr(L""); return;
        case ID_S_RRFILE: { if (v.busy || RrGameRunningSaid()) return; wstring p; if (Browse(5, p)) ConvertRr(p); return; }  // asked before the picker
        case ID_S_EXACT: gl.global.exactMode = !gl.global.exactMode; SaveSettings(); Invalidate(); return;  // no game's setting changes (rr-notes.md)
        case ID_S_RECONVERT: case ID_S_MODELFILE: {
            if (v.busy) return;  // an install may be copying the model file
            if (ModelGameRunningSaid()) return;  // a running game holds it: said before the search (and the picker)
            wstring p;
            if (id == ID_S_MODELFILE && !Browse(2, p)) return;
            const bool noScan = g.noScan;
            RunTask(T_MODEL, [p, noScan]() -> std::function<void()> {  // the search and the conversion off the window's thread
                wstring msg;
                bool ok = false, again = false;
                ConvertModelAgain(p, noScan, msg, ok, again);
                HealthMsg hm;
                ReadModelState(hm);  // the new model's state read here too (it hashes the file), not on the window's thread
                return [msg, ok, again, hm, chose = !p.empty()] {
                    TakeModelState(hm);
                    if (ok) { ShowToast(ModelConvertedWords(again, chose), 1); return; }
                    // not converted: a note in words (the search's own long text behind Show details)
                    // What to do follows the cause, as the Ray Reconstruction data's note (ConvertRr): a running game holding the
                    // data, a full disk, Windows refusing the folder, the file the player chose, else none found
                    const wstring stays = msg.find(L"model data in use stays") != wstring::npos ? L" The model data in use stays." : L"";
                    const wstring what = Has(msg, L"could not be moved") ? wstring(L"The current model data couldn't be moved, most likely because a running game uses it. "
                                                                                   L"Close your games and try again. Nothing changed.")
                                         : Has(msg, L"disk is full") || Has(msg, L"not enough space") ? L"The drive with your user folder is full. Free some space and try again." + stays
                                         : Has(msg, L"denied") ? L"Windows didn't allow WrongGPU to save the model data in your user folder. Try again; if it keeps happening, collect "
                                                                 L"the logs and report it." + stays
                                         : chose ? L"That file can't be used. Choose the DLSS file of a recent game (version 310.5 or newer)." + stays
                                                 : L"No DLSS file WrongGPU can use was found. Install or update a recent game with DLSS, or choose its DLSS file." + stays;
                    InformDetails(L"Model M couldn't be converted.\n\n" + what, kIconError, msg);  // the same title as when the task itself fails (RunTask)
                };
            });
            return;
        }
        case ID_S_ADDFOLDER: {
            wstring p;
            if (!Browse(1, p)) return;
            // a folder listed already (the same path, any case, with or without its last backslash) is not listed twice
            auto same = [](wstring a, wstring b) {
                a = FullPath(a); b = FullPath(b);
                while (a.size() > 3 && (a.back() == L'\\' || a.back() == L'/')) a.pop_back();
                while (b.size() > 3 && (b.back() == L'\\' || b.back() == L'/')) b.pop_back();
                return IEq(a, b);
            };
            for (auto& f : gl.global.extraFolders)
                if (same(f, p)) { ShowToast((BaseName(f).empty() ? f : BaseName(f)) + L" is in the list already.", 0); return; }  // its name, as the other folder toasts
            gl.global.extraFolders.push_back(p); SaveSettings(); addedFolder = p; StartScan(); Invalidate();
            return;
        }
        case ID_S_CLEARFOLDERS: {
            const size_t n = gl.global.extraFolders.size();
            // what leaves is what only the folders found (ScanGames lists the games with WrongGPU, the stores' and the games added
            // by hand whatever the folder list says)
            if (!Confirm(n == 1 ? L"Remove the added game folder from the list?\n\nGames found only in this folder leave the Library; games with WrongGPU stay. "
                                  L"Nothing on disk changes; add it again any time."
                                : L"Remove the " + std::to_wstring(n) + L" added game folders from the list?\n\nGames found only in these folders leave the Library; "
                                  L"games with WrongGPU stay. Nothing on disk changes; add them again any time.",
                         0, L"Remove", L"Cancel", true))
                return;
            gl.global.extraFolders.clear(); SaveSettings(); StartScan(); Invalidate(); return;
        }
        case ID_S_PLAY: gl.global.playButton = !gl.global.playButton; SaveSettings(); Invalidate(); return;
        case ID_S_UPDATE: gl.global.updateCheck = !gl.global.updateCheck; SaveSettings(); Invalidate(); return;  // an address being typed stays
        case ID_S_URL: FieldNow(); if (v.edUrl.h) SetFocus(v.edUrl.h); return;
        case ID_S_SAVEURL: SaveUrl(false); return;
        case ID_S_CHECKNOW: SaveUrl(true); return;
        case ID_S_SELFINSTALL: {
            if (v.busy) return;  // it adopts every game: not while an operation works in one
            HWND hwnd = v.hwnd;
            // the worker gets its own copy of the settings (the page stays live: Add folder, Play button... change gl.global
            // meanwhile) and the 0.5 installs the Library lists, adopted with the managed games
            const GlobalSettings gs = gl.global;
            const std::vector<wstring> adopt = AdoptableDirs();
            RunTask(T_SELFINSTALL, [hwnd, gs, adopt]() -> std::function<void()> {  // the install and the adoptions off the window's thread
                wstring why;
                std::vector<wstring> kept;
                if (!InstallLauncherForUser(L"", L"", hwnd, gs, adopt, why, &kept))
                    return [why] { InformDetails(L"WrongGPU couldn't be installed for this user.\n\n" + PlainTaskCause(why, kCauseLauncher), kIconError, why); };
                return [kept] {
                    SaveSettingsForInstalledCopy();
                    v.launcherInstalled = LauncherInstalled();
                    // a portable copy: its settings went with it, so its folder (the app, WrongGPU-data) is not needed any more
                    ShowToast(InstalledWords(kept, gl.paths.portable ? L"WrongGPU is installed. Open it from Start next time; you can delete this portable copy after you close it."
                                                                     : L"WrongGPU is installed. It's in Start and in Installed apps."),
                              kept.empty() ? 1 : 0);
                    StartScan();  // the games it adopted
                };
            });
            return;
        }
        case ID_S_SELFUNINSTALL: {
            if (!v.live || v.busy) return;  // it can remove WrongGPU from the games: not while an operation works in one
            if (!LauncherInstalled()) { v.launcherInstalled = false; Invalidate(); return; }  // nothing installed to uninstall (removed meanwhile)
            // destructive: a question with the mark, as a game's Uninstall (POLISH-AUDIT 19), and no default (Return does not
            // remove WrongGPU from every game unread). The games'
            // question only when a game has WrongGPU. Asked in a portable copy's window (Apps & features found it open): the
            // installed launcher goes, and this copy stays open
            // what it finds is read on a worker ("Preparing..."): every game's record and the drives they are on, the shared data
            // (a game on a network share that is offline holds each look for the share's timeout, never the window)
            auto look = std::make_shared<LauncherUninstallLook>();
            if (!WaitForRead(L"", kLookVerb, T_NONE, [look] { *look = LookBeforeLauncherUninstall(g.ctx); })) return;
            const bool anyGame = look->anyGame, here = !gl.paths.portable;
            // Collect logs from the question (uninstalling deletes the logs; opened from Installed apps, Cancel closes the window):
            // offered when the Library lists a game with WrongGPU; the question comes back after it
            const bool logs = anyGame && std::any_of(v.games.begin(), v.games.end(), [](const GameEntry& e) { return e.hasRecord; });
            int a = 0;
            for (;;) {
                a = RunDialog(0, UninstallLauncherText(*look, here, logs), UninstallLauncherButtons(anyGame, logs), 0);
                if (a != 3 || !v.live) break;
                CollectLogsBeforeUninstall();
            }
            // Installed apps' Uninstall opened this window only for the question: Cancel closes it again (nothing else was asked for)
            const bool justForThis = openedToUninstall;
            openedToUninstall = false;
            if (a == 0) { if (justForThis && v.hwnd) PostMessageW(v.hwnd, WM_CLOSE, 0, 0); return; }
            const bool games = a == 1;
            HWND hwnd = v.hwnd;
            RunTask(T_SELFUNINSTALL, [games, anyGame, here, hwnd]() -> std::function<void()> {  // it can restore every game (elevated where needed): off the window's thread
                wstring why;
                std::vector<wstring> offline;  // games on a drive that isn't connected: left as they are (SelfUninstall)
                Ui ui = QuietUi();
                if (!UninstallLauncher(games, L"", hwnd, ui, why, &offline))
                    return [why] { InformDetails(L"The launcher couldn't be uninstalled.\n\n" + PlainTaskCause(why, kCauseLauncherUninstall), kIconError, why); };
                return [games, anyGame, here, offline] {
                    std::vector<wstring> names;  // by the Library's names where it lists them, else by their drive
                    for (auto& d : offline) {
                        wstring n;
                        for (auto& e : v.games) if (n.empty() && IEq(FullPath(e.dir), FullPath(d))) n = e.name;
                        names.push_back(!n.empty() ? n : d.size() > 1 && d[1] == L':' ? L"A game on drive " + d.substr(0, 2) : wstring(L"A game on a network drive"));
                    }
                    // the games it did besides those (the Library's games with WrongGPU, before this): none, and no "other games"
                    bool others = false;
                    for (auto& e : v.games)
                        if (e.hasRecord && std::none_of(offline.begin(), offline.end(), [&e](const wstring& d) { return IEq(FullPath(e.dir), FullPath(d)); })) others = true;
                    Inform(LauncherUninstalledText(anyGame, games, here, names, others));
                    if (here) { DestroyWindow(v.hwnd); return; }
                    v.launcherInstalled = LauncherInstalled();
                    StartScan();  // the games it changed
                    StartHealth(false);  // the model and RR data went with it when no game used them: the status line and Settings follow
                    Invalidate();
                };
            });
            return;
        }
        case ID_S_REMOVEDATA: if (!v.busy) RemoveDataAsked(); return;  // ui_words.cpp: a portable copy's way out
        // first run
        // not while the checks run or Continue installs (its adoptions): one first-run worker at a time
        case ID_FR_MODEL: { if (v.first.busy || v.busy) return; wstring p; if (Browse(2, p)) { g.userDll = p; StartFirstRun(); } return; }
        case ID_FR_SEARCH: if (!v.first.busy && !v.busy) { g.userDll.clear(); StartFirstRun(); } return;  // one more look through the games
        case ID_FR_CHECKS: v.first.checksOpen = !v.first.checksOpen; Invalidate(); return;  // the passed checks, folded or shown
        // the install offer: a card chooses, Continue carries the choice out (nothing chosen: neither)
        case ID_FR_INSTALL: case ID_FR_PORTABLE:
            if (v.first.busy || v.busy) return;
            v.first.choice = id == ID_FR_INSTALL ? 1 : 2;
            Invalidate();
            return;
        case ID_FR_CONTINUE: {
            if (v.first.busy || v.busy) return;
            // Only now does anything on this PC change, as chosen: the launcher installed (or its data put next to it), the
            // model written, the 0.5 installs adopted (those in protected folders with one permission prompt)
            const bool install = v.first.installOffer && v.first.choice == 1;
            if (v.first.installOffer && v.first.choice == 2 && !FirstRunPortable()) return;
            const std::shared_ptr<ModelState> model = firstRunModel;
            std::vector<wstring> adopt = firstRunAdopt;
            for (auto& d : AdoptableDirs())  // a scan since the checks may list more
                if (std::find_if(adopt.begin(), adopt.end(), [&d](const wstring& x) { return IEq(x, d); }) == adopt.end()) adopt.push_back(d);
            if (!install && !model && adopt.empty()) { FinishFirstRun(); return; }
            HWND hwnd = v.hwnd;
            const GlobalSettings gs = gl.global;
            RunTask(T_FIRSTRUN, [hwnd, install, model, adopt, gs]() -> std::function<void()> {  // off the window's thread ("Installing...")
                wstring why, modelWhy;
                std::vector<wstring> kept;
                if (install && !InstallLauncherForUser(L"", L"", hwnd, gs, adopt, why, &kept))  // nothing else is done: the choice stays open
                    return [why] { InformDetails(L"WrongGPU couldn't be installed for this user.\n\n" + PlainTaskCause(why, kCauseLauncher), kIconError, why); };
                if (!install && !adopt.empty()) AdoptGames(adopt, hwnd, &kept);  // each keeps its own uninstaller (no launcher entry)
                if (model) {
                    try { LogLine(g.ctx, L"first run: Model M " + WriteModel(g.ctx, *model, NoUi()).outcome); }
                    catch (const SetupError& e) { modelWhy = e.msg; LogLine(g.ctx, L"first run: Model M was not written: " + e.msg); }
                }
                HealthMsg hm;
                if (model) ReadModelState(hm);  // the written model's state read here too (it hashes the file), not on the window's thread
                return [install, kept, modelWhy, model, hm] {
                    firstRunModel.reset();
                    firstRunAdopt.clear();
                    wstring installedExe;
                    if (install && windowLock && LauncherInstalled(&installedExe) && !IEq(FullPath(installedExe), FullPath(SelfPath()))) {
                        // as the replace does: the downloaded file never stays open as a launcher of its own. The first run is
                        // done in the settings the installed copy reads, which opens in this window's place
                        gl.global.firstRunDone = true;
                        SettleFirstRunLog();
                        SaveSettings();
                        SaveSettingsForInstalledCopy();
                        if (!modelWhy.empty())
                            InformDetails(L"Model M couldn't be converted.\n\nWrongGPU found a DLSS file but couldn't save the model data. Try Convert again in Settings, or "
                                          L"choose another DLSS file there.", kIconError, modelWhy);
                        Inform(InstalledWords(kept, L"WrongGPU is installed.\n\nThe installed WrongGPU opens now. Next time, open it from Start. You can delete this downloaded file."), 2);
                        HandOverToInstalled(*windowLock, installedExe, L"installed");
                        return;
                    }
                    if (install) {
                        SaveSettingsForInstalledCopy();
                        v.launcherInstalled = LauncherInstalled();
                        // this copy keeps running until it closes: the downloaded file can go after that
                        ShowToast(InstalledWords(kept, L"WrongGPU is installed. It's in Start and in Installed apps."), kept.empty() ? 1 : 0);
                    } else if (!kept.empty()) ShowToast(NotAdoptedWords(kept), 0);
                    FinishFirstRun();
                    if (model) TakeModelState(hm);
                    StartScan();  // the games it adopted
                    if (!modelWhy.empty())
                        InformDetails(L"Model M couldn't be converted.\n\nWrongGPU found a DLSS file but couldn't save the model data. Try Convert again in Settings, or choose "
                                      L"another DLSS file there.", kIconError, modelWhy);
                };
            });
            return;
        }
        default: return;
    }
}

}  // namespace lui
}  // namespace wg::app
