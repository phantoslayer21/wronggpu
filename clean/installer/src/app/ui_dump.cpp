// SPDX-License-Identifier: MIT
// WrongGPU launcher window: "WrongGPU.exe ui-dump", the window's pages painted offscreen as JSON (every page's widgets,
// the keyboard's paths the launcher tests check) and optional PNGs. Split from launcher_ui.cpp (the window, its input and
// operations), whose helpers it uses (launcher_window.h). Written for this project.
#include "app/launcher_window.h"

#include "wg_core/checks.h"
#include "wg_core/json_writer.h"
#include "wg_core/launcher.h"
#include "wg_core/logs.h"
#include "wg_core/model.h"
#include "wg_core/optiscaler.h"
#include "wg_core/payload.h"
#include "wg_core/rrconfig.h"

namespace wg::app {
using namespace wg::app::lui;

namespace win {
const wchar_t* RoleName(Role r) {
    switch (r) {
        case R_BUTTON: return L"Button"; case R_TOGGLE: return L"Switch"; case R_RADIO: return L"Segment"; case R_ITEM: return L"Item";
        case R_COMBO: return L"Select"; case R_LINK: return L"Link"; case R_MENUITEM: return L"MenuItem"; case R_EDIT: return L"Edit"; case R_SCROLL: return L"ScrollBar";
        case R_DIALOG: return L"Dialog"; case R_MENU: return L"Menu"; case R_PROGRESS: return L"Progress";
        default: return L"Text";
    }
}
void DumpHits(Json& j) {
    j.Arr();
    for (auto& h : v.hits) {
        if (!h.shown) continue;
        j.Obj().S("class", RoleName(h.role)).S("text", h.name).N("id", h.id).B("visible", true).B("enabled", h.enabled)
            .N("x", (long long)std::lround(h.r.x)).N("y", (long long)std::lround(h.r.y)).N("w", (long long)std::lround(h.r.w)).N("h", (long long)std::lround(h.r.h));
        if (h.state >= 0 && !Passive(h.role)) j.B("checked", h.state == 1);
        if (!h.value.empty()) j.S("value", h.value);
        if (!h.help.empty()) j.S("help", h.help);  // UI Automation's help text (a settings row's sentence)
        j.End();
    }
    j.End();
}
// The ui-dump's texts of the game page (moved here from launcher_ui.cpp, which is near its size limit)
wstring OptiInfoText(const OptiView& o) {  // the ui-dump's opti_info (the old tab's text)
    if (!o.have) return o.headline;
    wstring t = o.headline + L"\r\nManaged by WrongGPU (not changed here):";
    for (auto& r : o.locked) t += L" [" + FromUtf8(r.section) + L"] " + FromUtf8(r.key) + L"=" + FromUtf8(r.value) + L";";
    if (o.editable.empty()) t += L"\r\nThis OptiScaler build has none of the keys the launcher lets you change.";
    return t;
}
wstring OutsideText() {
    wstring t;
    for (auto& x : v.outside) t += (t.empty() ? L"Changed outside the launcher: " : L", ") + FromUtf8(x.key) + L" = " + FromUtf8(x.fileValue) + L" (profile: " + FromUtf8(x.profileValue) + L")";
    return t;
}
wstring OptiOutsideText() {
    wstring o;
    for (auto& x : v.opti.outside) o += (o.empty() ? L"Changed by OptiScaler (or by hand): " : L", ") + FromUtf8(x.key) + L" = " + FromUtf8(x.fileValue) + L" (ours: " + FromUtf8(x.ours) + L")";
    return o;
}

// The page's text for the ui-dump ("info"): status, last run, problems, the rules table, the proxy plan, leftovers.
wstring GameInfo(const GameEntry& e) {
    wstring t;
    if (e.hasRecord) {
        t += L"WrongGPU " + e.installedVersion + L" installed (this launcher has " + g.ctx.data->version + L")";
        if (e.hosted) t += L" \u00b7 OptiScaler as " + (e.proxy.empty() ? wstring(L"its proxy DLL") : e.proxy) + L" (WrongGPU's setup)";
        t += L"\r\n";
        if (e.hasPrevious) t += L"Roll back to " + e.previous + L" is possible\r\n";
        t += L"Last run: " + e.last.Summary() + L"\r\n";
        for (auto& n : e.last.notes) t += L"  ! " + n + L"\r\n";
        for (auto& w : e.last.warnings) t += L"  Warning: " + w.text + L"\r\n";
        if (e.last.found && e.hosted && !e.last.optiPlugin && e.last.ready) t += L"  Plugin: WrongGPU ran through OptiScaler's FidelityFX entry\r\n";
        for (auto& p : e.problems) t += L"Problem: " + p + L"\r\n";
        if (e.recordV1) t += L"Installed by setup 0.5.x: Repair or Update makes it a launcher install\r\n";
    } else {
        t += e.statusText + L"\r\n";
    }
    t += L"Install folder: " + e.dir + L"\r\n";
    for (auto& f : e.coexist) {
        if (f.decision == Decision::Managed) continue;
        const wchar_t* tag = f.decision == Decision::Refuse ? L"Refused" : f.decision == Decision::AskSwitch ? L"Asks first" : f.decision == Decision::Migrate ? L"Asks first"
                           : f.decision == Decision::Warn ? L"Warning" : f.decision == Decision::Chain ? L"Works next to it" : L"Found";
        t += wstring(tag) + L": " + f.what + L" - " + f.message + L"\r\n";
    }
    if (!e.hasRecord && !e.proxyPlan.name.empty() && e.status != GameStatus::Refused && e.status != GameStatus::ForeignMod)
        t += L"OptiScaler would be installed as " + e.proxyPlan.name + L" (" + e.proxyPlan.reason + L")\r\n";
    if (!e.leftovers.empty()) t += L"Leftover WrongGPU files (matched to WrongGPU " + e.leftovers[0].version + L" by their hashes): Clean up removes only these; Adopt makes them an install\r\n";
    if (!v.fixture && !IsFile(e.exe)) t += L"The game's exe is gone (moved or uninstalled): Remove files deletes what WrongGPU left in the folder.\r\n";
    return t;
}
// The permission prompts a test build has logged so far (WRONGGPU_TEST_ELEVATION_LOG, written by cli.cpp's
// StartElevated); 0 in a release build, which never logs them and reads no test variable.
static long long TestPromptCount() {
#ifdef WRONGGPU_TESTING
    string t;
    const wstring f = EnvVar(L"WRONGGPU_TEST_ELEVATION_LOG");
    if (f.empty() || !ReadAll(f, t)) return 0;
    long long n = 0;
    for (char c : t) n += c == '\n';
    return n;
#else
    return 0;
#endif
}
}  // namespace win
using namespace win;

// The window's pages painted offscreen (never shown, never activated): every page's widgets (role, name, id, place,
// state) as JSON, a game page for each listed game (with its "more" menu open), optional PNGs. No dialog, no install,
// no network, no HIP probe or model conversion. Development and the launcher tests.
int RunLauncherUiDump(const wstring& outFile, const wstring& shotDir, bool noScan, const wstring& apply) {
    g.noScan = noScan;
    CreateLauncherWindow(false);
    RefreshModelNow();
    SetGpu(DetectGpu());
    v.launcherInstalled = LauncherInstalled();
    Json j;
    j.Obj().S("version", g.ctx.data->version).S("build", g.ctx.data->buildId).S("payload", g.ctx.data->payloadName).S("data_dir", gl.paths.dataDir);
    auto shot = [&](const wchar_t* name) { return shotDir.empty() ? wstring() : Join(shotDir, name); };
    // the capsule's Apply, as the window runs it: the operation's worker, its result as WM_APP_OPDONE (the page read again)
    auto applyAndWait = [] {
        Activate(ID_G_APPLY);
        const ULONGLONG t0 = GetTickCount64();
        MSG m2;
        while (v.busy && GetTickCount64() - t0 < 120000) {
            if (PeekMessageW(&m2, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&m2); DispatchMessageW(&m2); }
            else Sleep(10);
        }
        return !v.busy;
    };
    // the PNG at the window's size; the widgets of the whole page (a frame as tall as the page)
    auto tall = [&] { RenderFrame(kDefaultW, std::max(760.f, v.contentH[v.page] + 20), 1, L""); };
    auto dumpPage = [&](const char* key, const wstring& title, const wchar_t* png) {
        RenderFrame(kDefaultW, 760, 1, shot(png));
        tall();
        j.Key(key).Obj().S("title", title).Key("controls");
        DumpHits(j);
        j.End();
    };
    // first run (synchronously: the worker thread's result is waited for with a message loop)
    v.first.installOffer = !LauncherInstalled() && !gl.paths.portable;
    v.first.choice = v.first.installOffer ? 1 : 0;
    StartFirstRun();
    MSG m;
    while (firstRunBusy && GetMessageW(&m, nullptr, 0, 0)) { TranslateMessage(&m); DispatchMessageW(&m); }
    ArtLoadNow([] { std::vector<wstring> ids; for (auto& e : v.games) ids.push_back(e.appid); return ids; }());
    dumpPage("first_run", L"Welcome to WrongGPU", L"first-run.png");
    wstring firstFocus;  // what has the keyboard once the checks are done (Continue)
    for (auto& h : v.hits) if (h.id == v.focus) firstFocus = h.name;
    v.first.open = false;  // as FinishFirstRun leaves it (without saving the settings or the release check)
    v.page = P_LIBRARY;
    v.focus = 0;
    focusFirstStop = true;
    dumpPage("list", L"Library", L"list.png");
    // the featured banner: 240 high in the default window (a full row of lockups under it), 300 in a taller one (UI review #10)
    j.Key("banner").Obj().N("default_h", (long long)LibraryBannerH(760 - kBarH)).N("tall_h", (long long)LibraryBannerH(1000 - kBarH)).End();
    {  // the keyboard on the Library: the first run's end puts the focus on its first stop; Down from a lockup reaches its
       // action and Up comes back; the toolbar's two tabs are one Tab stop
        j.Key("keyboard").Obj().S("first_run_focus", firstFocus);
        wstring after;
        for (auto& h : v.hits) if (h.id == v.focus && v.focus) after = h.name;
        j.S("after_first_run", after);
        int tile = 0;
        for (auto& h : v.hits)
            if (!tile && h.id >= ID_TILE && h.id < ID_TILE + kTileSpan)
                for (auto& a : v.hits) if (a.id == h.id + kTileSpan && a.enabled && a.focusable) tile = h.id;
        j.N("tile", tile);
        if (tile) {
            v.focus = tile; v.focusVisible = true;
            RenderFrame(kDefaultW, 760, 1, L"");
            OnKey(VK_DOWN, 0);
            j.B("down_action", v.focus == tile + kTileSpan);
            RenderFrame(kDefaultW, 760, 1, L"");
            OnKey(VK_UP, 0);
            j.B("up_lockup", v.focus == tile);
        }
        int navStops = 0;
        for (const Hit* f : Focusables()) navStops += f->id == ID_NAV_LIBRARY || f->id == ID_NAV_SETTINGS;
        j.N("nav_stops", navStops).End();
        v.focus = 0; v.focusVisible = false;
        v.target[P_LIBRARY] = v.scroll[P_LIBRARY] = 0;
    }
    {  // a search without a match: its "Clear search" from the keyboard puts the focus in the search field (the button goes
       // with the empty state), not on the first game's Play
        v.search = L"zz no game like this";
        RenderFrame(kDefaultW, 760, 1, L"");
        bool clear = false;
        for (auto& h : v.hits) clear |= h.id == ID_E_CLEAR && h.enabled;
        j.Key("clear_search").Obj().B("button", clear);
        if (clear) {
            v.focus = ID_E_CLEAR; v.focusVisible = true;
            Activate(ID_E_CLEAR);
            RenderFrame(kDefaultW, 760, 1, L"");
            j.B("cleared", v.search.empty()).B("on_search", v.focus == ID_SEARCH);
        }
        j.End();
        v.search.clear();
        v.focus = 0; v.focusVisible = false; v.editFocusId = 0;
        RenderFrame(kDefaultW, 760, 1, L"");
    }
    {  // back from the page of the featured banner's game (it has no lockup on the Library): the banner's Game settings has
       // the focus. A run on record makes an installed game the banner's.
        j.Key("featured_back").Obj();
        int gi = -1;
        for (int i = 0; i < (int)v.games.size() && gi < 0; ++i)
            if (v.games[i].hasRecord && !v.games[i].adoptable && v.games[i].status == GameStatus::Installed) gi = i;
        if (gi >= 0) {
            const LastRun keep = v.games[gi].last;
            v.games[gi].last.found = true;
            v.games[gi].last.when = L"2026-10-31 20:00";
            RenderFrame(kDefaultW, 760, 1, L"");
            const int feat = v.featIndex;
            j.N("feat", feat);
            if (feat >= 0) {
                v.focus = ID_FEAT0 + 2; v.focusVisible = true;
                RenderFrame(kDefaultW, 760, 1, L"");
                Activate(ID_FEAT0 + 2);  // its page
                RenderFrame(kDefaultW, 760, 1, L"");
                Navigate(P_LIBRARY);
                RenderFrame(kDefaultW, 760, 1, L"");
                j.B("focus_on_banner", v.focus == ID_FEAT0 + 2);
            }
            v.games[gi].last = keep;
            v.page = P_LIBRARY;
            v.focus = 0; v.focusVisible = false;
            v.target[P_LIBRARY] = v.scroll[P_LIBRARY] = 0;
        }
        j.End();
    }
    {  // the keyboard follows the game it pressed when an operation's result moves the Library under it (made-up games, copies
       // of an installed one): the banner that takes another game once its own Repair finished; a lockup's Update that made
       // its game the banner's; a lockup that moved to a shelf out of view (the page follows it)
        j.Key("place_focus").Obj();
        int gi = -1;
        for (int i = 0; i < (int)v.games.size() && gi < 0; ++i)
            if (v.games[i].hasRecord && !v.games[i].adoptable && v.games[i].status == GameStatus::Installed) gi = i;
        if (gi >= 0) {
            const std::vector<GameEntry> keepGames = v.games;
            GameEntry base = v.games[(size_t)gi];
            base.last.found = false;
            auto made = [&](const wstring& name, GameStatus st, bool ran) {
                GameEntry e = base;
                e.name = name; e.dir = Join(base.dir, name); e.exe = Join(e.dir, L"game.exe"); e.status = st;
                e.last.found = ran; e.last.when = ran ? L"2026-10-31 20:00" : L"";
                return e;
            };
            auto focusName = [] { for (auto& h : v.hits) if (h.id == v.focus) return h.name; return wstring(); };
            v.page = P_LIBRARY; v.pageChangedAt = 0;
            v.target[P_LIBRARY] = v.scroll[P_LIBRARY] = 0;
            // the banner shows Alpha (it needs a repair; nothing played yet); its Repair finishes and Bravo's update leads
            v.games = {made(L"Alpha", GameStatus::NeedsRepair, false), made(L"Bravo", GameStatus::UpdateAvailable, false)};
            v.focus = 0; v.focusVisible = true;
            RenderFrame(kDefaultW, 760, 1, L"");
            const int feat0 = v.featIndex;
            v.focus = ID_FEAT0 + 1;
            RenderFrame(kDefaultW, 760, 1, L"");
            const wstring before0 = focusName();
            v.games[0].status = GameStatus::Installed;
            RenderFrame(kDefaultW, 760, 1, L"");
            j.Key("banner_changed").Obj().N("feat_before", feat0).N("feat_after", v.featIndex).S("before", before0).S("after", focusName())
                .B("on_alpha", v.focus == ID_TILEQA || v.focus == ID_TILE).End();
            // Xray (played last) and Bravo both need an update; the banner shows Bravo; Xray's Update from its lockup makes it
            // the banner's game: the banner's Play has the focus, not the pill that slid into Xray's place
            v.games = {made(L"Bravo", GameStatus::UpdateAvailable, false), made(L"Xray", GameStatus::UpdateAvailable, true)};
            v.focus = 0;
            RenderFrame(kDefaultW, 760, 1, L"");
            const int feat1 = v.featIndex;
            v.focus = ID_TILEQA + 1;
            RenderFrame(kDefaultW, 760, 1, L"");
            const wstring before1 = focusName();
            v.games[1].status = GameStatus::Installed;
            RenderFrame(kDefaultW, 760, 1, L"");
            j.Key("became_banner").Obj().N("feat_before", feat1).N("feat_after", v.featIndex).S("before", before1).S("after", focusName())
                .B("on_banner_action", v.focus == ID_FEAT0 + 1).End();
            // a small window, a long shelf: Tango's Repair from "Needs attention" at the page's top; once repaired it moves to
            // the end of "With WrongGPU", out of view: the page scrolls to it
            v.games = {made(L"Played", GameStatus::Installed, true)};
            for (int i = 0; i < 30; ++i) { wchar_t n[16]; swprintf(n, 16, L"Copy %02d", i + 1); v.games.push_back(made(n, GameStatus::Installed, false)); }
            v.games.push_back(made(L"Tango", GameStatus::NeedsRepair, false));
            const int tango = (int)v.games.size() - 1;
            v.focus = 0;
            RenderFrame(760, 560, 1, L"");
            v.target[P_LIBRARY] = v.scroll[P_LIBRARY] = 0;
            v.focus = ID_TILEQA + tango;
            RenderFrame(760, 560, 1, L"");
            bool shownBefore = false;
            for (auto& h : v.hits) if (h.id == v.focus) shownBefore = h.shown;
            revealFocus = false;
            v.games[(size_t)tango].status = GameStatus::Installed;
            RenderFrame(760, 560, 1, L"");
            bool movedOut = false;
            for (auto& h : v.hits) if (h.id == v.focus) movedOut = !h.shown;
            const bool asked = revealFocus;
            if (revealFocus) {  // as the window's paint does after the frame
                revealFocus = false;
                for (auto& h : v.hits) if (h.id == v.focus) { const Hit t = h; FocusHit(&t); break; }
                v.scroll[P_LIBRARY] = v.target[P_LIBRARY];
                RenderFrame(760, 560, 1, L"");
            }
            bool shownAfter = false;
            for (auto& h : v.hits) if (h.id == v.focus) shownAfter = h.shown;
            j.Key("moved_shelf").Obj().B("shown_before", shownBefore).B("moved_out", movedOut).B("reveal", asked).B("same_game", v.focus == ID_TILEQA + tango)
                .B("shown_after", shownAfter).N("scroll", std::lround(v.scroll[P_LIBRARY])).End();
            v.games = keepGames;
            v.focus = 0; v.focusVisible = false; revealFocus = false;
            v.target[P_LIBRARY] = v.scroll[P_LIBRARY] = 0;
            RenderFrame(kDefaultW, 760, 1, L"");
        }
        j.End();
    }
    j.Key("rows").Arr();
    for (auto& e : v.games) j.Str(e.name + L" - " + e.statusText + (e.hasRecord ? L" - Last run: " + e.last.Summary() : L""));
    j.End();
    j.Key("games").Arr();
    for (size_t i = 0; i < v.games.size(); ++i) {
        ShowEntry(v.games[i], false);
        ArtLoadNow({v.cur.appid});
        // the Upscaler as the page opens (UI review #5): its Advanced rows folded unless one is set (AdvancedWanted); the
        // toggle shows them. The controls below are dumped with them shown
        tall();
        const bool advAtOpen = v.advOpen;
        bool advLink = false, runsOnShown = false;
        for (auto& h : v.hits) { advLink = advLink || h.id == ID_G_ADVMORE; runsOnShown = runsOnShown || h.id == ID_G_EXEC0; }
        if (advLink && !v.advOpen) Activate(ID_G_ADVMORE);
        RenderFrame(kDefaultW, 760, 1, L"");
        OpenMoreMenu();
        wchar_t name[32]; swprintf(name, 32, L"game-%zu.png", i);
        RenderFrame(kDefaultW, 760, 1, shot(name));
        tall();
        j.Obj().S("name", v.cur.name).S("appid", v.cur.appid).S("info", GameInfo(v.cur)).S("outside", OutsideText())
            .S("open_log", LastRunLog(v.cur.dir, v.cur.last));  // the log Open log opens (r6 #3)
        j.Key("advanced").Obj().B("open_at_start", advAtOpen).B("link", advLink).B("runs_on_at_start", runsOnShown).B("open_after_toggle", v.advOpen).End();
        j.Key("controls");
        DumpHits(j);
        j.S("opti_info", OptiInfoText(v.opti)).S("opti_outside", OptiOutsideText()).Key("opti_controls");
        DumpHits(j);
        v.menu.open = false;
        // Borderlands 4 / Hogwarts Legacy (UI review #2): the Engine.ini file check read again (Check again: its state and its
        // toast), and "Show me how" at the default size: the page scrolled to the card, the keyboard on the first step's Open
        // folder (rr_config_fix=off, as its Apply leaves it; nothing written)
        if (v.cur.hasRecord && RrConfigGameFor(v.cur.exe)) {
            const int atOpen = v.rrcfg;
            v.toast = Toast();
            RrConfigAction(ID_G_RRCFG_CHECK);
            j.Key("rrconfig").Obj().N("state_at_open", atOpen).N("state", v.rrcfg).S("file", v.rrcfgFile).S("check_toast", v.toast.text)
                .S("fix", FromUtf8(v.saved.get("rr_config_fix"))).S("question_text", RrConfigQuestionText(v.cur.dir, v.cur.exe))  // r4 #9
                .B("askable", RrConfigAskable(v.cur.dir, v.cur.exe)).B("rr_installed", v.cur.rrInstalled);  // r7 #5
            // Open folder before the game made its settings folder: nothing made, the toast says to start the game first (r2 #7;
            // never with a folder there: that would open File Explorer)
            if (v.rrcfgFile.empty() || !IsDir(Parent(v.rrcfgFile))) {
                v.toast = Toast();
                RrConfigAction(ID_G_RRCFG_OPEN);
                j.S("open_toast", v.toast.text).B("open_made_folder", !v.rrcfgFile.empty() && IsDir(Parent(v.rrcfgFile)));
            }
            const Profile keep = v.saved;
            SetSetting(v.saved, "rr_config_fix", "off", nullptr);
            v.edit = v.saved;
            v.scroll[P_GAME] = v.target[P_GAME] = 0;
            v.focus = 0; v.focusVisible = false; v.toast = Toast();
            ShowRrConfigSteps(v.cur.dir);
            RenderFrame(kDefaultW, 760, 1, L"");  // the steps drawn: the page scrolls to their card
            RenderFrame(kDefaultW, 760, 1, shot(L"rrconfig-show-me-how.png"));
            const Hit* open = nullptr;
            for (auto& h : v.hits) if (h.id == ID_G_RRCFG_OPEN) open = &h;
            j.Key("show_me_how").Obj().N("scroll", std::lround(v.scroll[P_GAME])).N("pin_h", std::lround(v.pinH)).B("focus_open", v.focus == ID_G_RRCFG_OPEN)
                .B("open_shown", open && open->shown && open->r.y >= v.pinH - 1 && open->r.b() <= 760 - GameBarH()).S("toast", v.toast.text)
                .B("pending", !v.rrcfgReveal.empty()).End();
            j.End();
            v.saved = v.edit = keep;
            v.scroll[P_GAME] = v.target[P_GAME] = 0;
            v.focus = 0; v.focusVisible = false; v.toast = Toast(); v.rrcfgReveal.clear();
            // r2b #1: Fix it for me chosen before the game's first start (nothing written, no line in the record) and the game's
            // settings folder there now: due. --apply=rrdue applies it as the window does when it opens or shows the page (once a
            // session: a second ask starts nothing)
            {
                const bool due = v.rrcfg >= 0 && !v.rrcfgFixed && v.saved.get("rr_config_fix") == "on";
                j.Key("rrconfig_due").Obj().B("fixed", v.rrcfgFixed).B("due", due).B("entry_due", v.cur.rrConfigDue).B("protected", v.cur.rrConfigProtected);
                // --apply=rrdue-library (r3 #1): the same apply with the Library shown (the launcher opening), not the game's page
                const bool fromLibrary = apply == L"rrdue-library";
                if ((apply == L"rrdue" || fromLibrary) && due) {
                    auto waitBusy = [] {
                        const ULONGLONG t0 = GetTickCount64();
                        MSG m2;
                        while (v.busy && GetTickCount64() - t0 < 120000) {
                            if (PeekMessageW(&m2, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&m2); DispatchMessageW(&m2); }
                            else Sleep(10);
                        }
                    };
                    // the card as drawn now: its "Ready to add" row and its button (r2c)
                    auto card = [&] {
                        tall();
                        wstring ready, fixb;
                        for (auto& h : v.hits) {
                            if (!h.shown) continue;
                            if (h.id == ID_G_RRCFG_FIX) fixb = h.name;
                            if (ready.empty() && h.name.rfind(L"Ready to add", 0) == 0) ready = h.name;
                        }
                        j.S("ready_text", ready).S("fix_button", fixb);
                    };
                    // the permission prompts asked so far (a test build lists them in WRONGGPU_TEST_ELEVATION_LOG: cli.cpp)
                    auto asks = [] { return TestPromptCount(); };
                    const wstring dir = v.cur.dir, exe = v.cur.exe;
                    // r3 #1: an unsaved change on the page (Sparkle guard) stays unsaved through the window's own apply (Engine.ini alone)
                    const string sgWas = v.edit.get("sparkle_guard"), sgNew = sgWas == "off" ? "on" : "off";
                    SetSetting(v.edit, "sparkle_guard", sgNew, nullptr);
                    auto editKept = [&] { return v.edit.get("sparkle_guard") == sgNew && v.saved.get("sparkle_guard") == sgWas && Dirty(); };
                    v.toast = Toast();
                    if (fromLibrary) {
                        // r3 #5: the Library as the launcher shows it: the game's tile (in "Needs attention" when its line waits for a click)
                        v.page = P_LIBRARY;
                        RenderFrame(kDefaultW, 760, 1, L"");
                        wstring pill, tile;
                        for (auto& h : v.hits) {
                            if (h.id >= ID_TILEQA && h.id < ID_TILEQA + kTileSpan && h.name.find(v.cur.name) != wstring::npos) pill = h.name;
                            else if (h.id >= ID_TILE && h.id < ID_TILE + kTileSpan && h.name.find(v.cur.name) != wstring::npos) tile = h.name;
                        }
                        wstring caption;  // the shelf's caption that counts it (r4 #8)
                        for (auto& h : v.hits) if (caption.empty() && h.name.find(L"Ray Reconstruction setting ready to add") != wstring::npos) caption = h.name;
                        bool red = true;  // r4 #8: this game alone never makes the tab's count red
                        for (auto& e : v.games) if (IEq(e.dir, v.cur.dir)) red = RedAttention(e);
                        j.Key("library").Obj().N("attention", NeedsAttentionCount()).S("pill", pill).S("tile", tile).B("red", red)
                            .B("ready", RrConfigReadyToAdd(v.cur)).S("caption", caption).End();
                    }
                    // r3 #7: an apply that can't start yet (the last session's list shown) isn't marked as tried: it starts once it can
                    v.listCached = true;
                    const bool blocked = ApplyRrConfigDueNow(dir, exe);
                    v.listCached = false;
                    j.B("blocked_started", blocked);
                    const bool started = ApplyRrConfigDueNow(dir, exe);
                    waitBusy();
                    const wstring toastAfter = v.toast.text;
                    if (fromLibrary) { v.page = P_GAME; RrConfigAction(ID_G_RRCFG_CHECK); }  // the page shown again: its file check read again
                    tall();
                    j.B("from_library", fromLibrary).B("started", started).B("applied", !v.busy).N("state", v.rrcfg).B("fixed_after", v.rrcfgFixed)
                        .S("toast", toastAfter).B("edit_kept", editKept()).B("again", ApplyRrConfigDueNow(dir, exe)).N("asks", asks());
                    v.toast = Toast();
                    // r2c: the game's folder is protected, so nothing was started (no permission prompt by itself). The card says
                    // "Ready to add" with Fix it now; the worker's own check (the entry's flag cleared as if it were stale, tried
                    // again: nothing run there either); then Fix it now, the player's click, and the elevated worker adds the line
                    if (!started && v.cur.rrConfigProtected && !fromLibrary) {
                        card();
                        ForgetRrConfigDue(dir);
                        v.cur.rrConfigProtected = false;
                        for (auto& e : v.games) if (IEq(e.dir, dir)) e.rrConfigProtected = false;
                        const bool s2 = ApplyRrConfigDueNow(dir, exe);
                        waitBusy();
                        j.Key("guard").Obj().B("started", s2).B("done", !v.busy).B("protected_after", v.cur.rrConfigProtected).N("state", v.rrcfg)
                            .B("fixed", v.rrcfgFixed).S("toast", v.toast.text).S("notice", v.notice).S("saved", FromUtf8(v.saved.get("rr_config_fix"))).N("asks", asks());
                        card();
                        j.End();
                        v.toast = Toast();
                        Activate(ID_G_RRCFG_FIX);
                        waitBusy();
                        tall();
                        j.Key("fix_now").Obj().B("done", !v.busy).N("state", v.rrcfg).B("fixed", v.rrcfgFixed).S("toast", v.toast.text).S("notice", v.notice)
                            .N("asks", asks()).B("edit_kept", editKept()).End();
                        v.toast = Toast();
                    }
                    SetSetting(v.edit, "sparkle_guard", sgWas, nullptr);
                }
                j.End();
            }
            // --apply=rrshowme (r3 #6): the question answered with Show me how: the steps show at once (Myself on the page), with no
            // permission prompt; where the folder needs none the choice is saved too, elsewhere it stays unsaved (Apply saves it)
            if (apply == L"rrshowme" && v.saved.get("rr_config_fix") == "ask") {
                auto prompts = [] { return TestPromptCount(); };
                const wstring dir = v.cur.dir, exe = v.cur.exe;
                v.toast = Toast(); v.notice.clear();
                v.scroll[P_GAME] = v.target[P_GAME] = 0;
                AnswerRrConfigQuestion(dir, exe, v.saved, 2);
                const bool busyStarted = v.busy;
                const bool editOff = v.edit.get("rr_config_fix") == "off";
                const ULONGLONG t0 = GetTickCount64();
                MSG m2;
                while (v.busy && GetTickCount64() - t0 < 120000) {
                    if (PeekMessageW(&m2, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&m2); DispatchMessageW(&m2); }
                    else Sleep(10);
                }
                RenderFrame(kDefaultW, 760, 1, L"");
                RenderFrame(kDefaultW, 760, 1, L"");
                bool steps = false;
                for (auto& h : v.hits) steps = steps || (h.id == ID_G_RRCFG_OPEN && h.shown);
                j.Key("show_me_how_answer").Obj().B("busy_started", busyStarted).B("edit_off_at_once", editOff).B("steps", steps)
                    .S("edit", FromUtf8(v.edit.get("rr_config_fix"))).S("saved", FromUtf8(v.saved.get("rr_config_fix"))).B("dirty", Dirty())
                    .B("focus_open", v.focus == ID_G_RRCFG_OPEN).S("toast", v.toast.text).S("notice", v.notice).N("asks", prompts()).End();
                v.toast = Toast(); v.edit = v.saved;
            }
            // --apply=rrfixforme (r4 #4): the question answered with Fix it for me, from a profile captured before a change made since
            // (here: its sparkle_guard flipped, as if the question had been asked before the player changed it): only rr_config_fix
            // is saved, the rest of the game's settings stay as they are now
            if (apply == L"rrfixforme" && v.saved.get("rr_config_fix") == "ask") {
                Profile stale = v.saved;
                SetSetting(stale, "sparkle_guard", stale.get("sparkle_guard") == "off" ? "on" : "off", nullptr);
                const wstring dir = v.cur.dir, exe = v.cur.exe;
                v.toast = Toast(); v.notice.clear();
                AnswerRrConfigQuestion(dir, exe, stale, 1);
                const bool busyStarted = v.busy;
                const ULONGLONG t0 = GetTickCount64();
                MSG m2;
                while (v.busy && GetTickCount64() - t0 < 120000) {
                    if (PeekMessageW(&m2, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&m2); DispatchMessageW(&m2); }
                    else Sleep(10);
                }
                const wstring toastAfter = v.toast.text;
                RrConfigAction(ID_G_RRCFG_CHECK);
                j.Key("fix_for_me_answer").Obj().B("busy_started", busyStarted).B("done", !v.busy).S("saved", FromUtf8(v.saved.get("rr_config_fix")))
                    .S("sparkle_guard", FromUtf8(v.saved.get("sparkle_guard"))).S("stale_sparkle_guard", FromUtf8(stale.get("sparkle_guard")))
                    .N("state", v.rrcfg).B("fixed", v.rrcfgFixed).S("toast", toastAfter).S("notice", v.notice).End();
                v.toast = Toast(); v.edit = v.saved;
            }
            // --apply=rrconfig (r2 #14): WrongGPU, then Myself, each chosen on the page and applied for real (the player's
            // Engine.ini written and taken back: test roots only in a test build), the page as read again after each
            if (apply == L"rrconfig") {
                j.Key("rrconfig_apply").Arr();
                for (int seg = 0; seg < 2; ++seg) {
                    tall();
                    Activate(ID_G_RRCFG0 + seg);
                    const bool dirty = Dirty();
                    const string want = v.edit.get("rr_config_fix");
                    v.toast = Toast();
                    const bool done = applyAndWait();
                    tall();
                    bool check = false, fix = false;
                    for (auto& h : v.hits) { check = check || h.id == ID_G_RRCFG_CHECK; fix = fix || h.id == ID_G_RRCFG_FIX; }
                    j.Obj().S("chose", want).B("dirty_before", dirty).B("applied", done).S("saved", v.saved.get("rr_config_fix")).N("state", v.rrcfg)
                        .B("dirty_after", Dirty()).B("check_button", check).B("fix_button", fix).B("fixed", v.rrcfgFixed).S("toast", v.toast.text).End();
                }
                j.End();
            }
        }
        j.End();
    }
    j.End();
    // the first game with WrongGPU, scrolled down with an unsaved change (all its OptiScaler settings shown), a frame
    // every 40 DIPs of the scroll range: the page's controls that take a click and its text fields that show (none under
    // the pinned plate or the unsaved bar); then Tab to a control hidden under the plate (it is scrolled out from under it)
    for (const GameEntry& ge : std::vector<GameEntry>(v.games)) {
        if (!ge.hasRecord) continue;
        ShowEntry(ge, false);
        ArtLoadNow({v.cur.appid});
        v.optiMore = v.advOpen = true;
        SetSetting(v.edit, "sparkle_guard", v.edit.get("sparkle_guard") == "off" ? "on" : "off", nullptr);
        const float fw = kDefaultW, fh = 560;
        RenderFrame(fw, fh, 1, L"");
        const float maxScroll = std::max(0.f, v.contentH[P_GAME] - fh);
        auto pageHit = [](const Hit& h) {
            return !ToolbarHit(h.id) && !(h.id >= ID_G_BACK && h.id <= ID_G_MORE) && h.id != ID_G_TITLE && h.id != ID_G_APPLY && h.id != ID_G_DISCARD && h.id != ID_SCROLLBAR &&
                   h.id != ID_G_STATUS && h.id != ID_G_PROGRESS && h.id != ID_G_UNSAVED;  // the plate's and the capsule's own words
        };
        j.Key("game_pinned").Obj().S("name", v.cur.name).B("dirty", Dirty()).N("height", (long long)fh).Key("frames").Arr();
        for (float sc = 0; sc < maxScroll + 40; sc += 40) {
            v.scroll[P_GAME] = v.target[P_GAME] = std::min(sc, maxScroll);
            RenderFrame(fw, fh, 1, L"");
            j.Obj().N("scroll", std::lround(v.scroll[P_GAME])).N("pin_h", std::lround(v.pinH)).N("bar_y", std::lround(fh - GameBarH())).Key("controls").Arr();
            for (auto& h : v.hits)
                if (h.shown && pageHit(h)) j.Obj().S("text", h.name).N("id", h.id).N("y", (long long)std::floor(h.r.y)).N("b", (long long)std::ceil(h.r.b())).End();
            j.End().Key("fields").Arr();
            for (auto& e : v.edOpti)
                if (e.want) j.Obj().B("shown", EditShown(e)).N("y", (long long)std::floor(e.r.y)).N("b", (long long)std::ceil(e.r.b())).End();
            j.End().End();
        }
        j.End();
        v.scroll[P_GAME] = v.target[P_GAME] = maxScroll;
        RenderFrame(fw, fh, 1, L"");
        Hit under;
        for (auto& h : v.hits) if (h.focusable && h.enabled && !h.shown && h.r.y < v.pinH && pageHit(h)) under = h;
        j.Key("tab_under_plate").Obj().N("id", under.id).N("pin_h_before", std::lround(v.pinH));
        if (under.id) {
            FocusHit(&under);
            v.scroll[P_GAME] = v.target[P_GAME];
            RenderFrame(fw, fh, 1, L"");
            const Hit* after = nullptr;
            for (auto& h : v.hits) if (h.id == under.id) after = &h;
            j.B("shown", after && after->shown).N("y", after ? (long long)std::floor(after->r.y) : -1).N("pin_h", std::lround(v.pinH));
        }
        j.End();
        // Tab into the right column (Last run, Game folder), which stops under the pinned header instead of scrolling with
        // the page: a control whose top is under the header comes out from under it (Reveal by where the column will be).
        // Shorter windows until the column is taller than its room.
        j.Key("tab_right_column").Obj();
        bool found = false;
        for (float rh0 : {560.f, 480.f, 420.f, 360.f}) {
            RenderFrame(fw, rh0, 1, L"");
            const float mx = std::max(0.f, v.contentH[P_GAME] - rh0);
            for (float sc = 0; sc < mx + 40 && !found; sc += 20) {
                v.scroll[P_GAME] = v.target[P_GAME] = std::min(sc, mx);
                RenderFrame(fw, rh0, 1, L"");
                Hit cand;
                for (auto& h : v.hits)
                    if (h.sticky && h.focusable && h.enabled && pageHit(h) && v.pinH > 0 && h.full.y < v.pinH - 1 && h.full.b() > v.pinH - 200) { cand = h; break; }
                if (!cand.id) continue;
                found = true;
                j.N("height", (long long)rh0).N("id", cand.id).N("scroll", std::lround(v.scroll[P_GAME]));
                FocusHit(&cand);
                v.scroll[P_GAME] = v.target[P_GAME];
                RenderFrame(fw, rh0, 1, L"");
                const Hit* after = nullptr;
                for (auto& h : v.hits) if (h.id == cand.id) after = &h;
                j.B("shown", after && after->shown).N("y", after ? (long long)std::floor(after->full.y) : -1).N("pin_h", std::lround(v.pinH));
            }
            if (found) break;
        }
        j.B("found", found).End();
        // the unsaved capsule closed by Discard from the keyboard: the focus goes back to the control that changed the
        // setting (Sparkle guard here), not to whatever stop was nearest the capsule
        v.edit = v.saved; v.focus = 0;
        v.scroll[P_GAME] = v.target[P_GAME] = 0;
        RenderFrame(fw, fh, 1, L"");
        Activate(ID_G_SPARKLE);
        v.focus = ID_G_DISCARD; v.focusVisible = true;
        RenderFrame(fw, fh, 1, L"");
        const bool capsule = Dirty();
        Activate(ID_G_DISCARD);
        RenderFrame(fw, fh, 1, L"");
        j.Key("capsule_focus").Obj().B("dirty_before", capsule).B("dirty_after", Dirty()).N("want", ID_G_SPARKLE).N("focus", v.focus).End();
        // the keyboard model: Space acts as it comes up, on the control it went down on (Tab away first: no press); a held
        // arrow in a segmented control steps once; Apply refusing an OptiScaler value puts the focus on that value's field;
        // a menu takes the focus to its control; a lone Alt opens no system-menu mode (Alt+Space still does); Browser Back
        j.Key("keys").Obj();
        v.edit = v.saved; v.optiMore = false;
        v.scroll[P_GAME] = v.target[P_GAME] = 0;
        RenderFrame(fw, fh, 1, L"");
        v.focus = ID_G_SPARKLE; v.focusVisible = true;
        OnKey(VK_SPACE, 0);
        const bool spaceDown = Dirty();
        SpaceUp();
        const bool spaceUp = Dirty();
        v.edit = v.saved;
        RenderFrame(fw, fh, 1, L"");
        v.focus = ID_G_SPARKLE;
        OnKey(VK_SPACE, 0);
        OnKey(VK_TAB, 0);
        SpaceUp();
        j.B("space_down_waits", !spaceDown).B("space_up_acts", spaceUp).B("space_moved_cancels", !Dirty());
        v.edit = v.saved;
        RenderFrame(fw, fh, 1, L"");
        int seg = 0;
        for (auto& h : v.hits) if (!seg && h.role == R_RADIO && h.enabled && h.focusable && h.state == 1) seg = h.id;
        j.N("segment", seg);
        if (seg) {
            v.focus = seg;
            OnKey(VK_RIGHT, 1 << 30);  // a repeat of a held arrow
            j.B("held_arrow_once", v.focus == seg && !Dirty());
            OnKey(VK_RIGHT, 0);
            j.B("arrow_steps", v.focus != seg);
            v.edit = v.saved;
        }
        int row = -1;
        for (size_t i = 0; i < v.opti.editable.size() && row < 0; ++i) {
            const OptiRow& r0 = v.opti.editable[i];
            string normal; wstring why;
            if (r0.def && (r0.def->kind == OptiKind::Number || r0.def->kind == OptiKind::Int) && !ValidateOptiValue(*r0.def, "not a number", &normal, &why)) row = (int)i;
        }
        j.N("opti_row", row);
        if (row >= 0 && (size_t)row < v.optiValues.size()) {
            const string keep = v.optiValues[(size_t)row];
            v.optiValues[(size_t)row] = "not a number";
            RenderFrame(fw, fh, 1, L"");
            v.focus = ID_G_APPLY;
            Activate(ID_G_APPLY);  // refused (its note: Inform), nothing written
            RenderFrame(fw, fh, 1, L"");
            bool shown = false;
            for (auto& h : v.hits) shown |= h.id == ID_G_OPTI0 + row * 10;
            j.B("apply_refused", !v.busy && Dirty()).B("bad_field_focus", v.focus == ID_G_OPTI0 + row * 10 && shown);
            v.optiValues[(size_t)row] = keep;
            v.optiMore = false;
        }
        v.focus = 0;
        RenderFrame(fw, fh, 1, L"");
        OpenMoreMenu();
        j.B("menu_focus", v.menu.open && v.focus == ID_G_MORE);
        v.menu.open = false;
        LRESULT res = 1;
        const bool alt = AccessMessage(WM_SYSCOMMAND, SC_KEYMENU, 0, res) && res == 0, altSpace = !AccessMessage(WM_SYSCOMMAND, SC_KEYMENU, ' ', res);
        j.B("lone_alt", alt).B("alt_space", altSpace);
        AccessMessage(WM_KEYDOWN, VK_BROWSER_BACK, 0, res);
        j.B("browser_back", v.page == P_LIBRARY).End();
        ShowEntry(ge, false);
        RenderFrame(fw, fh, 1, L"");
        j.End();
        v.focus = 0; v.focusVisible = false; v.optiMore = false; v.edit = v.saved;
        break;
    }
    // the game page's states from made-up runs and edits, on the first game with WrongGPU (UI review r2); nothing is written
    for (const GameEntry& ge : std::vector<GameEntry>(v.games)) {
        if (!ge.hasRecord) continue;
        auto texts = [&] {
            std::vector<wstring> t;
            tall();
            for (auto& h : v.hits) if (h.shown) t.push_back(h.name);
            return t;
        };
        auto find = [](const std::vector<wstring>& t, const wchar_t* part) { for (auto& x : t) if (x.find(part) != wstring::npos) return x; return wstring(); };
        j.Key("page_states").Obj();
        // #3: a run where the game asked for no upscaler, on OptiScaler's route and on the direct one: what to choose in the game
        for (int hosted = 1; hosted >= 0; --hosted) {
            ShowEntry(ge, false);
            v.cur.hosted = hosted != 0;
            LastRun& l = v.cur.last;
            l = LastRun();
            l.found = true; l.when = L"2026-11-05 20:00"; l.upscaler = L"network";
            const std::vector<wstring> t = texts();
            wstring useHelp;  // the Use WrongGPU switch's sentence (its UI Automation help text), by route (r2b #3)
            for (auto& h : v.hits) if (h.id == ID_G_MASTER) useHelp = h.help;
            j.Key(hosted ? "notused_hosted" : "notused_direct").Obj().S("warning", find(t, L"wasn't used in this run")).S("step2", find(t, L"Step 2"))
                .S("use_wronggpu", useHelp).End();
        }
        // #10: "Turn it on in the game" goes once Model M has upscaled a frame (the real parser on made-up logs): the runtime ready
        // alone (it logs that before any frame) keeps it
        {
            const string head = "[1] [x] ==== session start: game.exe (process 1) ====\r\n[1] [x] settings: upscaler=network\r\n"
                                "[2] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n";
            const string frame = "[3] [fe] #7 WrongGPU frame: R=1280x720 O=2560x1440 reset=1\r\n[3] [fe] #7 -> OK (WrongGPU)\r\n";
            j.Key("guide").Obj();
            for (int k = 0; k < 2; ++k) {
                ShowEntry(ge, false);
                v.cur.last = ParseLastRun(k ? head + frame : head);
                // the header's words from the same test (r2b #2): a healthy install, so only the last run decides
                v.cur.status = GameStatus::Installed; v.cur.adoptable = false; v.cur.problems.clear();
                const std::vector<wstring> t = texts();
                j.Key(k ? "upscaled" : "ready_only").Obj().B("shown", GuideShown()).B("heading", !find(t, L"Turn it on in the game").empty())
                    .S("last_run_sub", find(t, L"Started; no upscaled frame recorded in the log"))  // r3 #3, r4 #2: ready, no frame upscaled
                    .B("parsed_upscaled", v.cur.last.upscaled).B("parsed_ready", v.cur.last.ready).S("header_worked", find(t, L"Worked last time you played"))
                    .S("header_installed", find(t, L" is installed")).End();
            }
            j.End();
        }
        // r4 #1: a run of Ray Reconstruction alone (Borderlands 4, Hogwarts Legacy, RE Requiem: no front-end log), from the RR log's
        // lines through the launcher's own parse: a run (no "Not played yet"), the Denoiser row, the checklist gone, the header's Worked
        {
            const string rrLog = "00:31:46.071 runtime: instance 1 timing, 300 frames from frame 35701: GPU 1.594 ms per frame (pre 0.010, input 0.115), render 642x361, output 1920x1080\r\n"
                                 "00:31:48.974 runtime: destroy instance 1 after 36044 frames (0 failed)\r\n";
            ShowEntry(ge, false);
            v.cur.status = GameStatus::Installed; v.cur.adoptable = false; v.cur.problems.clear();
            LastRun l;
            ParseRrLog(rrLog, l);
            const bool counted = CountRrRun(l);
            l.when = L"2026-11-05 03:28";
            v.cur.last = l;
            // with WrongGPU's RR on (r5 #4: the header's second line then says Ray Reconstruction alone, not "Model M and ...")
            const bool rrWas = v.rrReady;
            v.rrReady = v.cur.rrUses = v.cur.rrInstalled = v.cur.rrEnabled = true;
            const std::vector<wstring> t = texts();
            // r6 #6: the Library's banner (this game, played last) follows the page header's rule: "Ray Reconstruction" alone
            wstring bannerSaid;
            {
                const std::vector<GameEntry> keepGames = v.games;
                for (auto& g2 : v.games) if (IEq(g2.dir, v.cur.dir)) g2 = v.cur;
                const Page pageWas = v.page;
                v.page = P_LIBRARY;
                for (auto& x : texts()) if (x.find(L"WrongGPU is on") != wstring::npos && x.find(v.cur.name) == wstring::npos) bannerSaid = x;
                v.page = pageWas;
                v.games = keepGames;
            }
            v.rrReady = rrWas;
            wstring denoiser;
            for (auto& x : t) if (x.rfind(L"Denoiser", 0) == 0) denoiser = x;
            j.Key("rr_only").Obj().B("counted", counted).B("rr_only", l.rrOnly).B("upscaled", l.upscaled).N("frames", l.rrFrames)
                .S("not_played", find(t, L"Not played yet")).S("denoiser", denoiser).S("upscaler_row", find(t, L"Upscaler: "))
                .B("guide", GuideShown()).S("step2", find(t, L"Step 2")).S("header_worked", find(t, L"Worked last time you played"))
                .S("size", find(t, L"Size: ")).B("right_gpu", RightGpu(v.cur)).S("banner", bannerSaid).End();
        }
        // r7 #6: what the Library tile of the game shown says to UI Automation (LockupStatus) with the page's last run (the game played
        // last leads the Library as its banner, so its tile isn't drawn here)
        auto tileSaid = [&] { return LockupSaid(v.cur); };
        // r5 #3: one game session in both logs (Cyberpunk 2077 with RR): RR's frames are never Model M's evidence (upscaled stays
        // false). r6 #1: where Model M upscaled none, RR did the upscaling: the Upscaler row names it, no "wasn't used" warning, no
        // late-start caveat, the checklist gone, the header's Worked with "Ray Reconstruction". Case A: Model M set, the game asked
        // it for nothing. Case B: a late start (the runtime ready, no frame of its own logged)
        {
            const string rrLog = "20:00:00.000 load: process 1500 attached C:\\Games\\Example\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                                 "20:19:59.000 runtime: instance 1 timing, 300 frames from frame 35701: GPU 3.579 ms per frame (pre 0.025), render 1280x720, output 1920x1080\r\n"
                                 "20:20:00.500 runtime: destroy instance 1 after 30962 frames (0 failed)\r\n";
            const string head = "[20:00:01.000 1500] [x] ==== session start: game.exe (process 1500) ====\r\n[20:00:01.100 1500] [x] settings: upscaler=network\r\n";
            for (int k = 0; k < 2; ++k) {
                ShowEntry(ge, false);
                v.cur.status = GameStatus::Installed; v.cur.adoptable = false; v.cur.problems.clear();
                v.cur.rrUses = v.cur.rrInstalled = true;
                LastRun l = ParseLastRun(k ? head + "[20:00:05.000 1500] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n" : head);
                ParseRrLog(rrLog, l);
                const bool counted = CountRrRun(l);
                l.when = L"2026-11-05 20:20";
                v.cur.last = l;
                const bool rrWas = v.rrReady;  // WrongGPU's RR on (the header's second line follows the last run: r5 #4, r6 #1)
                v.rrReady = v.cur.rrEnabled = true;
                const std::vector<wstring> t = texts();
                const wstring tile = tileSaid();  // r7 #6
                v.rrReady = rrWas;
                wstring denoiser;
                for (auto& x : t) if (x.rfind(L"Denoiser", 0) == 0) denoiser = x;
                j.Key(k ? "rr_mixed_late" : "rr_mixed_unused").Obj().S("tile", tile).B("counted", counted).B("rr_only", l.rrOnly).B("upscaled", l.upscaled).B("created", l.created)
                    .B("rr_upscaled", l.rrUpscaled).S("upscaler_row", find(t, L"Upscaler: ")).S("not_started", find(t, L"didn't start"))
                    .S("not_asked", find(t, L"didn't ask for an upscaler")).S("not_used", find(t, L"wasn't used")).S("choose_advice", find(t, L"choose DLSS as the upscaler"))
                    .S("late_sub", find(t, L"Started; no upscaled frame recorded in the log")).S("cant_confirm", find(t, L"can't confirm")).S("denoiser", denoiser)
                    .S("log_row", find(t, L"Log: ")).S("summary", l.Summary())
                    .S("header_worked", find(t, L"Worked last time you played")).S("header_installed", find(t, L" is installed")).B("guide", GuideShown()).End();
            }
        }
        // r6 #2: an RR attempt without a successful frame is a run that says so (never "Not played yet"), with the log's reason, and
        // no Worked: alone (k 0: no front-end session) and in one session with the front end where Model M wasn't used (k 1: the
        // Upscaler row says RR was asked for instead, r7 #1) or upscaled frames (k 2: Model M's run). r7 #2: each has the amber row
        // with the next step, and no green "No warnings"
        {
            const string rrLog = "21:00:00.000 load: process 4200 attached C:\\Games\\Example\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                                 "21:00:04.800 runtime: create instance 1: output 1920x1080, render 1280x720, mode 3\r\n"
                                 "21:00:04.810 runtime: instance 1 evaluate failed before any pass: runtime not ready\r\n"
                                 "21:00:04.811 rr: evaluate 1: runtime returned 13; colour passed through, success reported, reset on the next evaluate\r\n"
                                 "21:02:10.000 runtime: destroy instance 1 after 250 frames (250 failed)\r\n";
            for (int k = 0; k < 3; ++k) {
                ShowEntry(ge, false);
                v.cur.status = GameStatus::Installed; v.cur.adoptable = false; v.cur.problems.clear();
                v.cur.rrUses = v.cur.rrInstalled = true;
                const string fe = "[21:00:01.000 4200] [x] ==== session start: game.exe (process 4200) ====\r\n[21:00:01.100 4200] [x] settings: upscaler=network\r\n";
                LastRun l = k == 2 ? ParseLastRun(fe + "[21:00:02.000 4200] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n"
                                                      "[21:00:03.000 4200] [fe] #12 WrongGPU frame: R=1280x720 O=1920x1080\r\n[21:00:03.000 4200] [fe] #12 -> OK (WrongGPU)\r\n")
                          : k ? ParseLastRun(fe) : LastRun();
                ParseRrLog(rrLog, l);
                const bool counted = CountRrRun(l);
                l.when = L"2026-11-05 21:02";
                v.cur.last = l;
                const bool rrWas = v.rrReady;
                v.rrReady = v.cur.rrEnabled = true;
                const std::vector<wstring> t = texts();
                const wstring tile = tileSaid();  // r7 #6
                v.rrReady = rrWas;
                wstring denoiser;
                for (auto& x : t) if (x.rfind(L"Denoiser", 0) == 0) denoiser = x;
                j.Key(k == 2 ? "rr_failed_mixed_ok" : k ? "rr_failed_mixed" : "rr_failed_only").Obj().S("tile", tile).B("counted", counted).B("found", l.found).B("rr_only", l.rrOnly).B("upscaled", l.upscaled)
                    .B("rr_upscaled", l.rrUpscaled).S("not_played", find(t, L"Not played yet")).S("denoiser", denoiser).S("upscaler_row", find(t, L"Upscaler: "))
                    .S("model_m_not_used", find(t, L"Model M wasn't used")).S("wronggpu_not_used", find(t, L"WrongGPU wasn't used"))
                    .S("choose_advice", find(t, L"choose DLSS as the upscaler")).S("header_worked", find(t, L"Worked last time you played"))
                    .S("header_installed", find(t, L" is installed")).B("guide", GuideShown()).B("right_gpu", RightGpu(v.cur)).S("summary", l.Summary())
                    .S("step2", find(t, L"Step 2")).S("rr_warn", find(t, L"didn't finish a frame")).S("log_row", find(t, L"Log: ")).End();  // r7 #1, #2
            }
        }
        // r3 #3: step 3 follows the OptiScaler: WrongGPU's edition (its list names WrongGPU) says DLSS -> WrongGPU Model M; another
        // OptiScaler says AMD FSR and points at Last run, as Last run's foot does after a run where WrongGPU wasn't chosen by name
        {
            j.Key("step3_edition").Obj();
            for (int ed = 0; ed < 2; ++ed) {
                ShowEntry(ge, false);
                v.cur.optiEdition = ed != 0;
                v.cur.last = LastRun();
                j.S(ed ? "edition" : "other", find(texts(), L"Step 3"));
            }
            // r6 #4: before the first run with WrongGPU's Ray Reconstruction on, step 3 names the RR form too (the overlay's
            // DLSS -> WrongGPU RR, Last run's Ray Reconstruction)
            {
                ShowEntry(ge, false);
                v.cur.optiEdition = true; v.cur.hosted = true;
                v.cur.last = LastRun();
                const bool rrWas = v.rrReady;
                v.rrReady = v.cur.rrUses = v.cur.rrInstalled = v.cur.rrEnabled = true;
                j.S("edition_rr", find(texts(), L"Step 3")).B("rr_on", RrOn(v.cur));
                v.rrReady = rrWas;
            }
            // r5 #9: another OptiScaler after a late start (ready, not chosen by name): why it shows AMD FSR is said once, by Last
            // run's foot; step 3 keeps that it shows AMD FSR
            ShowEntry(ge, false);
            v.cur.optiEdition = false; v.cur.hosted = true;
            v.cur.status = GameStatus::Installed; v.cur.adoptable = false; v.cur.problems.clear();
            v.cur.last = LastRun();
            v.cur.last.found = v.cur.last.created = v.cur.last.ready = true; v.cur.last.upscaler = L"network"; v.cur.last.executor = L"dx12";
            v.cur.last.when = L"2026-11-05 20:00";
            const std::vector<wstring> t = texts();
            int byName = 0;
            for (auto& x : t) byName += x.find(L"doesn't list WrongGPU by name") != wstring::npos;
            j.S("other_after_run", find(t, L"Step 3")).N("by_name_count", byName);
            j.End();
        }
        // #14: the Upscaler's Advanced rows shown when a page opens (AdvancedWanted, as ShowEntry sets advOpen): a value off its
        // default (saved), an unsaved one, a value changed outside the launcher; folded at the defaults
        {
            auto opened = [&](const std::function<void()>& set) {
                ShowEntry(ge, false);
                set();
                v.advOpen = AdvancedWanted();
                tall();
                bool runsOn = false;
                for (auto& h : v.hits) runsOn = runsOn || (h.id == ID_G_EXEC0 && h.shown);
                return v.advOpen && runsOn;
            };
            j.Key("advanced").Obj().B("defaults", opened([] {}))
                .B("saved_off_default", opened([] { SetSetting(v.saved, "executor", "hip", nullptr); v.edit = v.saved; }))
                .B("unsaved", opened([] { SetSetting(v.edit, "sparkle_guard", v.edit.get("sparkle_guard") == "off" ? "on" : "off", nullptr); }))
                .B("outside", opened([] { v.outside.push_back({"output_filter", "analytic", "quality9"}); }))
                .End();
        }
        // #9: the checklist's "Turn on the FPS overlay": an unsaved On (step 3 says it turns on with Apply), Discard puts it back;
        // saved (as an Apply leaves the page), step 3 says it is on; an OptiScaler without the overlay's key: no promise, no
        // button. --apply: the button and the capsule's Apply for real (the game's files are written), then the page as read again
        {
            ShowEntry(ge, false);
            auto step3 = [&] { return find(texts(), L"Step 3"); };
            auto button = [&] { tall(); for (auto& h : v.hits) if (h.id == ID_G_GUIDEFPS && h.shown) return true; return false; };
            j.Key("fps").Obj().B("guide", GuideShown()).B("button", button()).S("step3", step3()).S("saved", FromUtf8(v.saved.get("fps_overlay")));
            Activate(ID_G_GUIDEFPS);
            j.S("after_press", FromUtf8(v.edit.get("fps_overlay"))).B("dirty", Dirty()).S("step3_pending", step3()).B("button_pending", button());
            Activate(ID_G_DISCARD);
            j.S("after_discard", FromUtf8(v.edit.get("fps_overlay"))).B("dirty_after_discard", Dirty()).B("button_after_discard", button());
            Activate(ID_G_GUIDEFPS);
            v.saved = v.edit;
            j.S("step3_saved", step3()).B("button_saved", button());
            ShowEntry(ge, false);
            for (size_t i = 0; i < v.opti.locked.size(); ++i)
                if (v.opti.locked[i].section == "Menu" && v.opti.locked[i].key == "ShowFps") { v.opti.locked.erase(v.opti.locked.begin() + (long long)i); break; }
            j.S("step3_nokey", step3()).B("button_nokey", button());
            if (apply == L"fps") {
                ShowEntry(ge, false);
                tall();
                Activate(ID_G_GUIDEFPS);
                const bool done = applyAndWait();
                j.B("applied", done).S("saved_after_apply", FromUtf8(v.saved.get("fps_overlay"))).B("dirty_after_apply", Dirty()).S("step3_applied", step3())
                    .B("button_applied", button());
            }
            j.End();
        }
        j.End();
        ShowEntry(ge, false);
        v.page = P_LIBRARY;
        break;
    }
    v.page = P_SETTINGS;
    v.urlText = gl.global.updateUrl;
    dumpPage("settings", L"Settings", L"settings.png");
    {  // Settings' section index: one Tab stop (the section in view); Down goes to the next entry, Up comes back
        int stops = 0, stop = 0;
        RenderFrame(kDefaultW, 760, 1, L"");
        for (const Hit* f : Focusables()) if (f->id >= ID_S_INDEX0 && f->id < ID_S_INDEX0 + 3) { ++stops; stop = f->id; }
        j.Key("index_keys").Obj().N("stops", stops).N("stop", stop);
        if (stop == ID_S_INDEX0) {
            v.focus = stop; v.focusVisible = true;
            RenderFrame(kDefaultW, 760, 1, L"");
            OnKey(VK_DOWN, 0);
            const int down = v.focus;
            RenderFrame(kDefaultW, 760, 1, L"");
            int stopNow = 0;
            for (const Hit* f : Focusables()) if (f->id >= ID_S_INDEX0 && f->id < ID_S_INDEX0 + 3) stopNow = f->id;
            OnKey(VK_UP, 0);
            j.B("down_next", down == ID_S_INDEX0 + 1).B("stop_follows", stopNow == ID_S_INDEX0 + 1).B("up_back", v.focus == ID_S_INDEX0);
            OnKey(VK_UP, 0);
            j.B("up_stays", v.focus == ID_S_INDEX0);
            // Return on "Launcher": its first stop takes the focus, which for Appearance is the chosen segment (Light here),
            // as Tab lands on it, not "System"
            const wstring keep = gl.global.appearance;
            gl.global.appearance = L"light";  // drawn chosen; the theme itself is not changed
            v.focus = ID_S_INDEX0 + 2;
            RenderFrame(kDefaultW, 760, 1, L"");
            Activate(ID_S_INDEX0 + 2);
            j.B("launcher_chosen_segment", v.focus == ID_S_THEME0 + 1);
            gl.global.appearance = keep;
            v.target[P_SETTINGS] = v.scroll[P_SETTINGS] = 0;
        }
        j.End();
        v.focus = 0; v.focusVisible = false;
        v.target[P_SETTINGS] = v.scroll[P_SETTINGS] = 0;
    }
    {  // a button that goes busy as it is pressed (Model data's, "Converting..."): the keyboard focus stays on it, shown off
       // with its busy name (a second Return does nothing, never another action), and Tab goes on from its place
        int id = 0;
        for (auto& h : v.hits) if ((h.id == ID_S_MODELMORE || h.id == ID_S_MODELFILE) && h.focusable && h.enabled) id = h.id;
        j.Key("busy_focus").Obj().N("id", id);
        if (id) {
            v.target[P_SETTINGS] = v.scroll[P_SETTINGS] = 0;
            v.focus = id; v.focusVisible = true;
            RenderFrame(kDefaultW, 760, 1, L"");
            v.busy = true; v.task = T_MODEL;  // as RunTask leaves it (no worker here)
            RenderFrame(kDefaultW, 760, 1, L"");
            const Hit* on = nullptr;
            int at = -1;
            for (size_t i = 0; i < v.hits.size(); ++i) if (v.hits[i].id == v.focus) { on = &v.hits[i]; at = (int)i; }
            j.B("kept", v.focus == id).S("name", on ? on->name : wstring()).B("enabled", on && on->enabled);
            OnKey(VK_RETURN, 0);  // nothing to press: no menu, no dialog, no other control
            j.B("return_inert", v.focus == id && !v.menu.open && !v.dialog.open);
            OnKey(VK_TAB, 0);
            int next = -1;
            for (size_t i = 0; i < v.hits.size(); ++i) if (v.hits[i].id == v.focus) next = (int)i;
            j.B("tab_on", at >= 0 && next > at);
            v.busy = false; v.task = T_NONE;
            v.focus = 0; v.focusVisible = false;
        }
        j.End();
    }
    j.End();
    DestroyWindow(v.hwnd);
    const string s = j.str() + "\n";
    try { WriteFileAtomic(outFile, s.data(), s.size()); } catch (const SetupError&) { return EXIT_FAILED; }
    return EXIT_OK;
}
}  // namespace wg::app
