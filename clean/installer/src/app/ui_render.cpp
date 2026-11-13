// SPDX-License-Identifier: MIT
// WrongGPU launcher window: the offscreen frames of "WrongGPU.exe ui-render" (development screenshots of the window, light
// and dark, at several sizes and scales) from a small fixture file of game records (tests/ui_fixture.ini): nothing is shown
// and no game folder is read; art comes from this PC's Steam cache. Every page, state, dialog and motion the window has a
// frame for. Split from launcher_ui.cpp (the window, its input and operations), whose helpers it uses (launcher_window.h).
// Written for this project.
#include "app/launcher_window.h"

#include "wg_core/json_writer.h"
#include "wg_core/rrconfig.h"
#include "wg_core/logs.h"
#include "wg_core/model.h"
#include "wg_core/optiscaler.h"
#include "wg_core/payload.h"
#include "wg_core/pe.h"

namespace wg::app {
using namespace wg::app::lui;
using namespace wg::app::win;

namespace {
// ---- fixtures (ui-render): game records from a small text file; no game folder is read ---------------------------------------
GameStatus ParseStatus(const wstring& s) {
    for (GameStatus x : {GameStatus::Eligible, GameStatus::Installed, GameStatus::UpdateAvailable, GameStatus::NeedsRepair, GameStatus::TurnedOff,
                         GameStatus::PlayersOptiScaler, GameStatus::ForeignMod, GameStatus::OldInstall, GameStatus::Orphaned, GameStatus::Refused,
                         GameStatus::Running, GameStatus::Leftovers, GameStatus::NeedsPrereq})
        if (IEq(s, StatusName(x))) return x;
    return GameStatus::Eligible;
}
struct FixtureGame { GameEntry e; Profile profile; OptiView opti; std::vector<OutsideEdit> outside; };
std::vector<FixtureGame> fixtures;
std::vector<std::pair<int, wstring>> fixtureFirstRun;
bool fixtureRrReady = false, fixtureRrBuild = false;  // [global] rr=ready, rrbuild=on: every frame starts from them

bool LoadFixture(const wstring& path, wstring* why) {
    string raw;
    if (!ReadAll(path, raw)) { *why = L"cannot read " + path; return false; }
    const wstring t = FromUtf8(raw);
    FixtureGame* cur = nullptr;
    wstring section;
    size_t pos = 0;
    while (pos < t.size()) {
        size_t nl = t.find(L'\n', pos);
        wstring line = t.substr(pos, nl == wstring::npos ? wstring::npos : nl - pos);
        pos = nl == wstring::npos ? t.size() : nl + 1;
        while (!line.empty() && (line.back() == L'\r' || line.back() == L' ')) line.pop_back();
        if (line.empty() || line[0] == L';' || line[0] == L'#') continue;
        if (line[0] == L'[') {
            section = Lower(line.substr(1, line.find(L']') - 1));
            if (section == L"game") { fixtures.emplace_back(); cur = &fixtures.back(); cur->profile = DefaultProfile(); cur->e.last = LastRun(); }
            continue;
        }
        const size_t eq = line.find(L'=');
        if (eq == wstring::npos) continue;
        const wstring k = Lower(line.substr(0, eq)), val = line.substr(eq + 1);
        if (section == L"global") {
            if (k == L"model") v.modelReady = val == L"ready";
            else if (k == L"gpu") { GpuInfo gi; gi.name = val; gi.amd = true; gi.rdna4 = val.find(L"RX 90") != wstring::npos; SetHealth(v.modelReady, gi); }
            else if (k == L"play") gl.global.playButton = val == L"on";
            else if (k == L"opti") gl.global.optiPackage = val;
            else if (k == L"update_check") gl.global.updateCheck = val == L"on";
            else if (k == L"update_url") gl.global.updateUrl = val;
            else if (k == L"hip") v.fixtureHip = val;
            else if (k == L"rr") v.rrReady = fixtureRrReady = val == L"ready";  // the Ray Reconstruction data
            else if (k == L"rrbuild") v.rrBuild = fixtureRrBuild = val == L"on";  // this build has the RR files
            else if (k == L"rr.source") v.rrSource = val;  // the RR model file it was converted from
            continue;
        }
        if (section == L"firstrun") { fixtureFirstRun.push_back({k == L"ok" ? 0 : k == L"problem" ? 2 : 1, val}); continue; }
        if (!cur) continue;
        GameEntry& e = cur->e;
        auto split = [](const wstring& s, wchar_t c) { std::vector<wstring> o; size_t a = 0; for (;;) { size_t b = s.find(c, a); o.push_back(s.substr(a, b == wstring::npos ? wstring::npos : b - a)); if (b == wstring::npos) break; a = b + 1; } return o; };
        if (k == L"name") e.name = val;
        else if (k == L"appid") e.appid = val;
        else if (k == L"source") e.source = val;
        else if (k == L"exe") { e.exe = val; if (e.dir.empty()) e.dir = Parent(val); }
        else if (k == L"dir") e.dir = val;
        else if (k == L"status") e.status = ParseStatus(val);
        else if (k == L"statustext") e.statusText = val;
        else if (k == L"version") { e.installedVersion = val; e.hasRecord = true; }
        else if (k == L"hosted") e.hosted = val == L"1";
        else if (k == L"proxy") e.proxy = val;
        else if (k == L"opti.edition") e.optiEdition = val == L"1";  // WrongGPU's edition of OptiScaler (r3 #3)
        else if (k == L"problem") e.problems.push_back(val);
        else if (k == L"lastrun") { e.last.found = true; e.last.when = val; e.last.upscaler = L"network"; e.last.created = true; }
        else if (k == L"executor") { e.last.executor = val; e.last.ready = e.last.upscaled = true; }  // a run where Model M worked
        else if (k == L"render") e.last.renderSize = val;
        else if (k == L"output") e.last.outputSize = val;
        else if (k == L"note") e.last.notes.push_back(val);
        else if (k == L"warning") { auto p = split(val, L'|'); e.last.warnings.push_back({p[0], p.size() > 1 ? p[1] : L""}); }
        else if (k == L"finding") {
            auto p = split(val, L'|');
            Finding f; ParseDecision(p[0], f.decision); f.what = p.size() > 1 ? p[1] : L""; f.message = p.size() > 2 ? p[2] : L"";
            if (p.size() > 3) ParseToolKey(ToUtf8(p[3]), f.tool);  // the tool (fingerprint.h's key): the page's words for it
            if (p.size() > 4) f.files = split(p[4], L',');
            e.coexist.push_back(f);
        }
        else if (k == L"plan") { auto p = split(val, L'|'); e.proxyPlan.name = p[0]; e.proxyPlan.reason = p.size() > 1 ? p[1] : L""; }
        else if (k.rfind(L"profile.", 0) == 0) SetSetting(cur->profile, ToUtf8(k.substr(8)), ToUtf8(val), nullptr);
        else if (k == L"rr") { e.rrUses = true; e.rrWhy = L"bin\\x64\\nvngx_dlssd.dll"; e.rrInstalled = val == L"installed"; }  // uses | installed (DLSS)
        else if (k == L"rr.enabled") { e.rrEnabled = val != L"off"; SetSetting(cur->profile, "rr_enabled", ToUtf8(val), nullptr); }
        else if (k == L"rr.mode") { e.rrMode = val; SetSetting(cur->profile, "rr_mode", ToUtf8(val), nullptr); }
        else if (k == L"rr.used") { e.last.rrFound = true; e.last.rrUsed = val == L"1"; e.last.rrFrames = e.last.rrUsed ? 1 : 0; }  // the last run's RR log (its frames: r6 #2)
        else if (k == L"rr.lastmode") e.last.rrMode = val;
        else if (k == L"rr.ms") e.last.rrMs = val;
        else if (k == L"opti.version") { cur->opti.have = true; cur->opti.version = val; }
        else if (k == L"opti.folder") cur->opti.folder = val;
        else if (k == L"opti.key" || k == L"opti.locked") {  // Section.Key=value
            const size_t e2 = val.find(L'=');
            OptiRow r;
            string sec, key;
            SplitOptiName(ToUtf8(val.substr(0, e2)), sec, key);
            r.section = sec; r.key = key; r.value = ToUtf8(val.substr(e2 + 1));
            r.locked = k == L"opti.locked";
            r.def = FindOptiKey(sec, key);
            (r.locked ? cur->opti.locked : cur->opti.editable).push_back(r);
        }
        else if (k == L"outside") { auto p = split(val, L'|'); if (p.size() >= 3) cur->outside.push_back({ToUtf8(p[0]), ToUtf8(p[1]), ToUtf8(p[2])}); }
    }
    const OptiView* firstOpti = nullptr;
    for (auto& f : fixtures) if (f.opti.have && !firstOpti) firstOpti = &f.opti;
    for (auto& f : fixtures) {
        GameEntry& e = f.e;
        SplitRoutineNotes(e.last);  // as ParseLastRun does: a ready run's start-up notes are routine (UI review #8)
        // a run with WrongGPU's edition of OptiScaler went through its list by name (today's real runs log the custom slot's lines):
        // so the page never says both "switch it in OptiScaler's menu" and "this OptiScaler doesn't list WrongGPU by name" (r4 #12)
        if (e.optiEdition && e.last.found && e.last.ready) e.last.optiPlugin = true;
        if (e.hasRecord && !f.opti.have && firstOpti) { const OptiView o = *firstOpti; f.opti = o; f.opti.folder.clear(); }
        if (e.statusText.empty()) {
            switch (e.status) {
                case GameStatus::Installed: e.statusText = L"Installed " + e.installedVersion; break;
                case GameStatus::UpdateAvailable: e.statusText = L"Installed " + e.installedVersion + L" \u2192 update " + g.ctx.data->version; break;
                case GameStatus::NeedsRepair: e.statusText = L"Needs repair (" + (e.problems.empty() ? wstring(L"changed files") : e.problems[0]) + L")"; break;
                case GameStatus::Eligible: e.statusText = L"Ready to install (tested)"; break;
                default: e.statusText = StatusName(e.status); break;
            }
        }
        if (e.hasRecord) e.hosted = true;
        f.opti.proxy = e.proxy;
        if (f.opti.folder.empty()) f.opti.folder = e.dir;
        v.games.push_back(e);
    }
    return true;
}
void ShowFixtureGame(size_t i) {
    v.saved = fixtures[i].profile;
    v.opti = fixtures[i].opti;
    v.outside = fixtures[i].outside;
    ShowEntry(fixtures[i].e, false);
}

}  // namespace

int RunLauncherUiRender(const wstring& outDir, const wstring& fixtureFile, int w, int h, const std::vector<float>& scales, const wstring& frames,
                        const wstring& theme, wstring* why) {
    CreateLauncherWindow(false);
    SetTheme(theme == L"dark");  // renders: light unless --theme=dark (the PC's own theme is not followed)
    v.fixture = true;
    gl.paths.portable = true;
    if (!LoadFixture(fixtureFile, why)) return EXIT_USAGE;
    gl.global.appearance = theme == L"dark" ? L"dark" : L"light";  // Settings' Appearance shows the theme the frames are drawn in
    std::vector<wstring> ids;
    for (auto& e : v.games) ids.push_back(e.appid);
    ids.push_back(L"3764200");  // RE Requiem's art (its frames below; not one of the Library's games)
    const std::vector<GameEntry> all = v.games;
    // the fixture games the pages show: the first one, and the first that matches
    auto find = [&](const std::function<bool(const GameEntry&)>& f) { for (size_t i = 0; i < fixtures.size(); ++i) if (f(fixtures[i].e)) return (int)i; return -1; };
    const int gUpdate = find([](const GameEntry& e) { return e.status == GameStatus::UpdateAvailable; });
    const int gRepair = find([](const GameEntry& e) { return e.status == GameStatus::NeedsRepair && e.last.found == false; });
    const int gNoArt = find([](const GameEntry& e) { return e.appid.empty() && e.status == GameStatus::Eligible; });
    const int gRefused = find([](const GameEntry& e) { return e.status == GameStatus::Refused; });
    const int gEligible = find([](const GameEntry& e) { return !e.appid.empty() && e.status == GameStatus::Eligible; });
    int gLong = -1;
    for (size_t i = 0; i < fixtures.size(); ++i) if (gLong < 0 || fixtures[i].e.name.size() > fixtures[(size_t)gLong].e.name.size()) gLong = (int)i;
    const wstring want = L"," + frames + L",";
    auto reset = [&] {
        v.games = all;
        v.page = P_LIBRARY; v.search.clear(); v.scanning = false; v.listCached = false; v.attnAll = false;
        for (int p = 0; p < P_COUNT; ++p) v.scroll[p] = v.target[p] = 0;
        v.menu.open = false; v.dialog = Dialog(); v.first = FirstRun(); v.toast = Toast();
        v.hot = v.focus = 0; v.focusVisible = false; v.anim.clear();
        v.busy = false; v.busyDir.clear(); v.step.clear(); v.busyVerb.clear(); v.progress = 0; v.queueTotal = v.queueDone = 0; v.queueDirs.clear();
        v.task = T_NONE;
        v.rrReady = fixtureRrReady; v.rrBuild = fixtureRrBuild; v.rrEdition = true; gl.global.exactMode = false; v.fixtureRrPrepared.clear();
        v.detailsOpen = v.lockedOpen = v.optiMore = v.advOpen = false; v.notice.clear(); v.noticeDir.clear();
        v.editFocusId = 0; v.checkingUpdate = false; v.optiBad = View::OptiBad();
        v.motionAt = -1;
        v.mouseX = v.mouseY = -1; v.wipeAt.clear(); v.wipeOut.clear();
        v.flipDir.clear(); v.flipAt = 0; v.flipT = -1; v.artHide.clear();
        if (CurrentAccent() != A_GREEN) { SetAccent(A_GREEN, WindowsAccentRgb()); SetTheme(theme == L"dark"); }
        if (pal.hc) { ContrastPreview(0); SetTheme(theme == L"dark"); }
    };
    // a Contrast theme's colours (Night sky with the dark frames, Desert with the light ones), whatever the PC uses
    auto contrast = [&] { ContrastPreview(theme == L"dark" ? 1 : 2); SetTheme(theme == L"dark"); };
    auto accent = [&](int a) { SetAccent(a, WindowsAccentRgb()); SetTheme(theme == L"dark"); };
    int failed = 0;
    for (float s : scales) {
        ArtScale(s);  // the art decoded at its drawn size for this scale
        ArtLoadNow(ids);
        const wstring tag = L"@" + std::to_wstring((int)std::lround(s * 100));
        auto frame = [&](const wchar_t* name, const std::function<void()>& setup) {
            if (!frames.empty() && want.find(L"," + wstring(name) + L",") == wstring::npos) return;
            reset();
            setup();
            failed += !RenderFrame((float)w, (float)h, s, Join(outDir, wstring(name) + tag + L".png"));
        };
        auto game = [&](int i) { if (i >= 0) ShowFixtureGame((size_t)i); };
        // the Library (its featured banner: Cyberpunk 2077, the game played last with WrongGPU on)
        frame(L"library", [] {});
        // the banner's other cases: no game played yet (the first that needs a repair leads, in red); no WrongGPU game at all
        // (no banner); a banner and tiles without hero art (header.jpg for Ratchet & Clank, a plate for Bodycam and the banner)
        frame(L"library-feature-attention", [] { for (auto& e : v.games) if (e.status == GameStatus::Installed) e.last = LastRun(); });
        // its Repair running from the banner: the banner's button says "Repairing..." (disabled), as the game page's does
        frame(L"library-feature-busy", [] {
            for (auto& e : v.games) if (e.status == GameStatus::Installed) e.last = LastRun();
            for (auto& e : v.games) if (e.status == GameStatus::NeedsRepair) { v.busyDir = e.dir; break; }
            v.busy = true; v.busyVerb = L"Repairing…"; v.step = L"Copying WrongGPU's files (3 of 7)";
        });
        // Update all running with the banner's game still to come: its line says it waits, the progress bar's track after
        // it (as its lockup would); the game being updated shows its bar on its lockup
        frame(L"library-feature-queue", [] {
            for (auto& e : v.games) {
                if (e.status == GameStatus::Installed) e.last = LastRun();
                if (e.status == GameStatus::NeedsRepair) { e.status = GameStatus::Installed; e.problems.clear(); e.last = LastRun(); }
            }
            for (auto& e : v.games) if (e.status == GameStatus::UpdateAvailable) v.queueDirs.push_back(e.dir);
            if (v.queueDirs.size() > 1) v.busyDir = v.queueDirs[1];
            v.busy = true; v.queueTotal = (int)v.queueDirs.size(); v.busyVerb = L"Updating…"; v.step = L"Copying WrongGPU's files (3 of 7)"; v.progress = 0.3f;
        });
        // the last session's list at start-up, while the scan reads the games again: the operations wait (Play works)
        frame(L"library-cached", [] { v.listCached = true; v.scanning = true; });
        frame(L"library-nobanner", [] {
            std::vector<GameEntry> none;
            for (auto& e : v.games) if (!e.hasRecord) none.push_back(e);
            v.games = none;
        });
        frame(L"library-fallbacks", [] { v.artHide[L"1895880"] = 1; v.artHide[L"2406770"] = 7; v.artHide[L"870780"] = 1; v.artHide[L"1091500"] = 7; });
        frame(L"library-blue", [&] { accent(A_BLUE); });
        frame(L"library-graphite", [&] { accent(A_GRAPHITE); });
        // the pointer on a piece of art (the middle of what the last pass drew of it): its upscale wipe
        auto pointAt = [&](int id) {
            RenderFrame((float)w, (float)h, s, L"");
            for (auto& a : v.wipeArts) if (a.id == id) { v.mouseX = a.r.cx(); v.mouseY = a.r.cy(); return; }
            for (auto& t : v.hits) if (t.id == id) { v.mouseX = t.r.cx(); v.mouseY = t.r.y + 60; return; }
        };
        // scrolled so a hit is at a part of the view (0 its top, 1 its bottom), measured in a first pass; none when it fits
        // (its whole rectangle: a hit under the fold is clipped to the view)
        auto scrollTo = [&](int id, float at) {
            RenderFrame((float)w, (float)h, s, L"");
            for (auto& t : v.hits)
                if (t.id == id) { v.scroll[P_LIBRARY] = v.target[P_LIBRARY] = std::max(0.f, std::round(t.full.b() - kBarH - ((float)h - kBarH) * at)); break; }
        };
        // the tile in view at every size (scrolled to it where it is under the fold): its upscale wipe, its art's ring. The
        // game's tile, else (behind "Show 2 more" in a small window) the first tile on the page with Steam art
        auto shownTile = [&](int id) {
            RenderFrame((float)w, (float)h, s, L"");
            int first = 0;
            for (auto& t : v.hits) {
                if (t.id == id) return id;
                const int i = t.id - ID_TILE;
                if (!first && i >= 0 && i < (int)v.games.size() && i < kTileSpan && !v.games[(size_t)i].appid.empty()) first = t.id;
            }
            return first ? first : id;
        };
        frame(L"library-hover", [&] { const int id = shownTile(ID_TILE + 1); scrollTo(id, 0.85f); v.hot = id; v.anim[id] = 1; pointAt(id); });
        frame(L"library-focus", [&] { const int id = shownTile(ID_TILE + 2); scrollTo(id, 0.85f); v.focus = id; v.focusVisible = true; v.anim[id] = 1; });
        frame(L"library-banner-focus", [] { v.focus = ID_FEAT0 + 2; v.focusVisible = true; });  // the ring on the banner's art (both themes: the art's colour)
        // the first repair's pill (its action, the second stop of a lockup): the keyboard's ring, the pointer's tint
        auto firstRepair = []() -> int { for (size_t i = 0; i < v.games.size(); ++i) if (v.games[i].status == GameStatus::NeedsRepair) return ID_TILEQA + (int)i; return ID_TILEQA; };
        frame(L"library-pill-focus", [&] { const int id = firstRepair(); v.focus = id; v.focusVisible = true; scrollTo(id, 0.85f); });
        frame(L"library-pill-hover", [&] { const int id = firstRepair(); v.hot = id; scrollTo(id, 0.85f); });
        // Borderlands 4 in a protected folder, its Engine.ini line ready to add on a click (r3 #5): in "Needs attention" with Fix it
        // now, and the toast the launcher shows once when it opens
        frame(L"library-rrconfig-ready", [&] {
            int bl = -1;
            for (size_t i = 0; i < v.games.size(); ++i)
                if (v.games[i].name == L"Borderlands 4") {
                    bl = (int)i;
                    GameEntry& e = v.games[i];
                    e.status = GameStatus::Installed; e.installedVersion = Ctx().data->version; e.statusText.clear();
                    e.rrConfigDue = e.rrConfigProtected = true;
                    e.rrUses = e.rrInstalled = true;  // WrongGPU's RR files in (r6 #5: only then is the line listed)
                }
            if (bl >= 0) scrollTo(ID_TILE + bl, 0.85f);
            ShowToast(ReadyToAddToast(), 0);
        });
        // ...the game played last, so the banner's: its Fix it now there (r3 #5)
        frame(L"library-rrconfig-ready-banner", [&] {
            for (auto& e : v.games)
                if (e.name == L"Borderlands 4") {
                    e.status = GameStatus::Installed; e.installedVersion = Ctx().data->version; e.statusText.clear();
                    e.rrConfigDue = e.rrConfigProtected = true; e.last.when = L"2026-11-05 10:00";
                    e.rrUses = e.rrInstalled = true;  // (r6 #5)
                }
        });
        // r6 #6: Borderlands 4 played last, a run of Ray Reconstruction alone: the banner says Ray Reconstruction, as the page header does
        frame(L"library-rronly-banner", [&] {
            for (auto& e : v.games)
                if (e.name == L"Borderlands 4") {
                    e.status = GameStatus::Installed; e.installedVersion = Ctx().data->version; e.statusText.clear(); e.problems.clear();
                    e.rrUses = e.rrInstalled = e.rrEnabled = true;
                    e.last = LastRun();
                    e.last.rrFound = e.last.rrUsed = true; e.last.rrFrames = 36044; e.last.rrMs = L"1.6"; e.last.rrMode = L"accurate";
                    CountRrRun(e.last);
                    e.last.when = L"2026-11-05 10:00";
                }
        });
        // r4 #8: that line the only attention item (the other games healthy): the tab's count is not red. r5 #5: RR's own switch
        // decides: the upscaler turned off in Borderlands 4 (RR on) keeps its line in Needs attention; RR turned off takes it out
        auto rrcfgLibrary = [&](GameStatus st, bool alone, bool rrOn = true, bool rrFiles = true) {
            int bl = -1;
            for (size_t i = 0; i < v.games.size(); ++i) {
                GameEntry& e = v.games[i];
                if (e.name == L"Borderlands 4") {
                    bl = (int)i;
                    e.status = st; e.installedVersion = Ctx().data->version; e.statusText.clear();
                    e.rrConfigDue = e.rrConfigProtected = true;
                    e.rrEnabled = rrOn;
                    e.rrUses = e.rrInstalled = rrFiles;  // (r6 #5: without WrongGPU's RR files the line is never listed)
                } else if (alone) {
                    if (e.hasRecord && e.status != GameStatus::TurnedOff && e.status != GameStatus::Running) {
                        e.status = GameStatus::Installed; e.installedVersion = Ctx().data->version; e.statusText.clear(); e.problems.clear();
                    } else if (e.status == GameStatus::Leftovers || e.status == GameStatus::Orphaned) e.status = GameStatus::Eligible;
                    e.adoptable = false;
                }
            }
            if (bl >= 0) scrollTo(ID_TILE + bl, 0.85f);
        };
        frame(L"library-rrconfig-ready-only", [&] { rrcfgLibrary(GameStatus::Installed, true); });
        frame(L"library-rrconfig-ready-off", [&] { rrcfgLibrary(GameStatus::TurnedOff, true); });
        frame(L"library-rrconfig-ready-rroff", [&] { rrcfgLibrary(GameStatus::Installed, true, false); });
        // r6 #5: the same Borderlands 4 install without WrongGPU's RR files (offline, or not the WrongGPU edition): not listed
        frame(L"library-rrconfig-norr", [&] { rrcfgLibrary(GameStatus::Installed, true, true, false); });
        // long lines under the art, cut with an ellipsis (the tooltip has them whole): an update from a long version; a
        // game in a folder with an older WrongGPU and one added by you with its own OptiScaler (on Available, scrolled to)
        auto longLines = [] {
            bool upd = false;
            int avail = 0;
            for (auto& e : v.games)
                if (!upd && e.status == GameStatus::UpdateAvailable) { e.installedVersion = L"0.4.5-rc12"; upd = true; }
                else if (avail < 2 && e.status == GameStatus::Eligible) {
                    e.status = avail ? GameStatus::PlayersOptiScaler : GameStatus::OldInstall;
                    e.source = avail ? L"Added" : L"";
                    e.appid.clear();
                    e.exe = L"C:\\Games\\" + e.name + L"\\game.exe";
                    ++avail;
                }
        };
        frame(L"library-lines", [&] { longLines(); });
        frame(L"library-lines-available", [&] { longLines(); scrollTo(ID_SHELF0 + 2, 0.1f); });
        // a folder's copy of the first game (its namesake has WrongGPU) with its own OptiScaler, on Available: the line's
        // words whole, before where it comes from
        frame(L"library-namesake", [&] {
            GameEntry c = all[0];
            c.hasRecord = false; c.adoptable = false; c.status = GameStatus::PlayersOptiScaler; c.statusText.clear();
            c.installedVersion.clear(); c.problems.clear(); c.last = LastRun(); c.appid.clear(); c.source.clear();
            c.dir = L"K:\\Games\\" + c.name; c.exe = c.dir + L"\\bin\\x64\\game.exe";
            v.games.push_back(c);
            scrollTo(ID_SHELF0 + 2, 0.1f);
        });
        frame(L"library-scrolled", [] { v.scroll[P_LIBRARY] = v.target[P_LIBRARY] = 5000; });
        frame(L"library-search", [] { v.search = L"wit"; });
        frame(L"library-nomatch", [] { v.search = L"zelda"; });
        frame(L"library-empty", [] { v.games.clear(); });
        frame(L"library-scanning", [] { v.games.clear(); v.scanning = true; });
        frame(L"library-task", [] { v.busy = true; v.task = T_MODEL; });  // a job that is not a game's: the line under the title says why every action waits
        frame(L"library-busy", [&] {  // Update all running, its first game done (it keeps its place): the head's progress, the bay's step
            for (auto& e : v.games) if (e.status == GameStatus::UpdateAvailable) v.queueDirs.push_back(e.dir);
            for (auto& e : v.games)
                if (v.queueDirs.size() > 1 && IEq(e.dir, v.queueDirs[0])) { e.status = GameStatus::Installed; e.installedVersion = g.ctx.data->version; }
                else if (v.queueDirs.size() > 1 && IEq(e.dir, v.queueDirs[1])) v.busyDir = e.dir;
            v.busy = true; v.queueTotal = (int)v.queueDirs.size(); v.queueDone = 1; v.step = L"Copying WrongGPU's files (3 of 7)"; v.progress = 0.4f;
        });
        frame(L"library-progress", [] {  // one game updating (62 %): its lockup shows the bar and "Updating... 62%" where its Update was
            int n = 0;
            for (auto& e : v.games) if (e.status == GameStatus::UpdateAvailable && n++ == 1) v.busyDir = e.dir;
            v.busy = true; v.busyVerb = L"Updating…"; v.step = L"Copying WrongGPU's files (3 of 7)"; v.progress = 0.62f;
        });
        frame(L"library-repair-update", [] {  // a repair that updates too (an older install): the lockup's short verb, "Repairing... 62%"
            // both repairs are of an older WrongGPU: the other one's pill says "Repair and update" (off while this one runs)
            for (auto& e : v.games) if (e.status == GameStatus::NeedsRepair) e.installedVersion = L"0.5.0";
            for (auto& e : v.games) if (e.status == GameStatus::NeedsRepair) { v.busyDir = e.dir; break; }
            v.busy = true; v.busyVerb = L"Repairing and updating…"; v.step = L"Copying WrongGPU's files (3 of 7)"; v.progress = 0.62f;
        });
        frame(L"library-cleanup", [] {  // a game that is gone and one with leftovers: after the repairs, before the updates, with Review
            bool gone = false, left = false;
            for (auto& e : v.games)
                if (!gone && e.status == GameStatus::UpdateAvailable) { e.status = GameStatus::Orphaned; gone = true; }
                else if (!left && e.status == GameStatus::Eligible) { e.status = GameStatus::Leftovers; left = true; }
            v.attnAll = true;  // the whole group: both in view
        });
        frame(L"library-repairs", [] {  // only repairs need attention: the group's sentence, no Update all
            for (auto& e : v.games) if (e.status == GameStatus::UpdateAvailable) { e.status = GameStatus::Installed; e.installedVersion = g.ctx.data->version; }
        });
        frame(L"library-adopt", [] {  // an install set up outside the launcher (WrongGPU setup 0.5): Adopt on its lockup
            for (auto& e : v.games) if (e.status == GameStatus::Installed) { e.adoptable = true; break; }
        });
        frame(L"library-calm", [] {  // nothing needs attention: no attention group, the games with WrongGPU first
            for (auto& e : v.games)
                if (e.status == GameStatus::UpdateAvailable || e.status == GameStatus::NeedsRepair) {
                    e.status = GameStatus::Installed; e.installedVersion = g.ctx.data->version; e.problems.clear();
                }
        });
        auto many = [&] {  // a big library: the fixture games again and again
            std::vector<GameEntry> lots;
            for (int k = 0; k < 6; ++k) for (auto e : all) { if (k) { e.name += L" (" + std::to_wstring(k + 1) + L")"; e.dir += std::to_wstring(k); } lots.push_back(e); }
            v.games = lots;
        };
        frame(L"library-many", [&] { many(); });
        // "Needs attention" longer than two rows of lockups (a big library): "Show 28 more" under it, keyboard focus, hover,
        // and opened whole with "Show fewer" (scrolled to it)
        frame(L"library-more-focus", [&] { many(); v.focus = ID_ATTNMORE; v.focusVisible = true; scrollTo(ID_ATTNMORE, 0.6f); });
        frame(L"library-more-hover", [&] { many(); v.hot = ID_ATTNMORE; scrollTo(ID_ATTNMORE, 0.6f); });
        frame(L"library-attention", [&] { many(); v.attnAll = true; v.scroll[P_LIBRARY] = v.target[P_LIBRARY] = 1700; });
        frame(L"library-many-scrolled", [&] { many(); v.scroll[P_LIBRARY] = v.target[P_LIBRARY] = 2600; });
        // a game's page (Cyberpunk 2077: installed and worked last time, so its page says RightGPU)
        frame(L"game", [&] { game(0); });
        frame(L"game-hero-focus", [&] { game(0); v.focus = ID_G_MORE; v.focusVisible = true; });  // the ring on the hero's art
        // "< Library" at the page's top: its ring whole over the toolbar's edge (FocusRingOverBar)
        frame(L"game-back-focus", [&] { game(0); v.focus = ID_G_BACK; v.focusVisible = true; });
        frame(L"game-back-focus-noart", [&] { v.artHide[L"1091500"] = 1; game(0); v.focus = ID_G_BACK; v.focusVisible = true; });
        frame(L"game-blue", [&] { accent(A_BLUE); game(0); });
        frame(L"game-graphite", [&] { accent(A_GRAPHITE); game(0); });
        frame(L"game-hero-noart", [&] { v.artHide[L"1091500"] = 1; game(0); });  // no hero: the cover and title header
        // RightGPU's flip as an install finishes on the game's page (no run yet): before, the bar half green, the letters
        // half way, after; the install-finished frame with its toast
        auto flip = [&](float t) { game(0); v.cur.last = LastRun(); v.flipDir = v.cur.dir; v.flipT = t; };
        frame(L"rightgpu-0", [&] { flip(0); });
        frame(L"rightgpu-fill", [&] { flip((120.f + 62.f) / 620.f); });  // the bar half green (ease-out)
        frame(L"rightgpu-cross", [&] { flip((420.f + 100.f) / 620.f); });
        frame(L"rightgpu-1", [&] { flip(1); });
        frame(L"rightgpu-installed", [&] { flip(0.1f); v.toast.text = L"WrongGPU is installed in " + v.cur.name + L"."; v.toast.tone = 1; });
        frame(L"game-menu", [&] { game(0); OpenMoreMenu(); });
        // More while another game's update runs: the items that wait, and why in a note at its end
        frame(L"game-menu-busy", [&] {
            game(0);
            if (gUpdate >= 0) { v.busy = true; v.busyDir = v.games[(size_t)gUpdate].dir; v.busyVerb = L"Updating\u2026"; v.step = v.busyVerb; }
            OpenMoreMenu();
        });
        // the page shown before its files are read (a slow disk): the header, and a quiet row for the settings
        frame(L"game-reading", [&] { game(0); ShowReadingPage(v.cur, v.curIndex); });
        frame(L"game-cover", [&] { game(0); pointAt(ID_G_COVER); });  // the pointer on the cover: its upscale wipe
        frame(L"game-menukey", [&] {  // the menu key's list (21 keys): fits the window
            game(0);
            v.scroll[P_GAME] = v.target[P_GAME] = 260;
            RenderFrame((float)w, (float)h, s, L"");
            auto key = [] { for (auto& x : v.hits) if (x.id >= ID_G_OPTI0 && x.id < ID_G_OPTI_END && x.role == R_COMBO) return x; return Hit{}; };
            const Hit k = key();
            if (k.id && (!k.shown || k.r.b() > v.H - 24)) {  // scrolled into view first, as it is when someone clicks it
                v.scroll[P_GAME] = v.target[P_GAME] = std::max(0.f, v.scroll[P_GAME] + k.r.cy() - v.H * 0.55f);
                RenderFrame((float)w, (float)h, s, L"");
            }
            if (const Hit k2 = key(); k2.id) Activate(k2.id);
        });
        frame(L"game-filter-menu", [&] {  // the output filter's choices, each with its line under the name
            game(0);
            v.scroll[P_GAME] = v.target[P_GAME] = 120;
            RenderFrame((float)w, (float)h, s, L"");
            bool there = false;
            for (auto& x : v.hits) there = there || x.id == ID_G_FILTER;
            if (there) Activate(ID_G_FILTER);
        });
        frame(L"game-unsaved", [&] {
            game(0);
            SetSetting(v.edit, "sparkle_guard", "on", nullptr);
            SetSetting(v.edit, "executor", "dx12", nullptr);
            v.scroll[P_GAME] = v.target[P_GAME] = 220;
        });
        frame(L"game-pinned", [&] { game(0); v.scroll[P_GAME] = v.target[P_GAME] = 560; });  // the plate under the window's top edge
        // the name just under the toolbar: the compact header has pinned, the header's art scrolls on under it (no second Play)
        frame(L"game-pinning", [&] { game(0); v.scroll[P_GAME] = v.target[P_GAME] = 185; });
        frame(L"game-pinning-noart", [&] { v.artHide[L"1091500"] = 1; game(0); v.scroll[P_GAME] = v.target[P_GAME] = 80; });
        frame(L"game-pinned-busy", [&] {
            game(gUpdate); v.scroll[P_GAME] = v.target[P_GAME] = 560;
            v.busy = true; v.busyDir = v.cur.dir; v.busyVerb = L"Updating…"; v.step = L"Copying WrongGPU's files (3 of 7)"; v.progress = 0.4f;
        });
        frame(L"game-bottom", [&] { game(0); v.detailsOpen = v.lockedOpen = v.optiMore = true; v.scroll[P_GAME] = v.target[P_GAME] = 5000; });
        frame(L"game-unsaved-bottom", [&] {  // the end of the page with the bar docked: the thumb stops above it
            game(0); v.optiMore = true;
            SetSetting(v.edit, "sparkle_guard", "on", nullptr);
            v.scroll[P_GAME] = v.target[P_GAME] = 5000;
        });
        frame(L"game-opti-invalid", [&] {  // Apply refused a number: the field keeps the keyboard, its red edge until it changes
            game(0); v.optiMore = true;
            for (size_t i = 0; i < v.opti.editable.size() && i < v.optiValues.size(); ++i) {
                const OptiKind kind = v.opti.editable[i].def ? v.opti.editable[i].def->kind : OptiKind::Number;
                if (kind != OptiKind::Number && kind != OptiKind::Int) continue;
                v.optiValues[i] = "fast";
                v.optiBad = View::OptiBad{v.cur.dir, i, "fast", L"Needs a number, such as 1.0, or Auto."};
                v.focus = ID_G_OPTI0 + (int)i * 10; v.focusVisible = true;
                break;
            }
            RenderFrame((float)w, (float)h, s, L"");  // its place on the page, then the page scrolled to show it (as revealFocus does)
            for (auto& x : v.hits)
                if (x.id == v.focus) v.scroll[P_GAME] = v.target[P_GAME] = std::max(0.f, x.full.y + x.scroll - v.H * 0.45f);
        });
        frame(L"game-applied", [&] {  // Apply's result: a toast (a success is always one), here with the header pinned
            game(0); v.scroll[P_GAME] = v.target[P_GAME] = 560;
            v.toast.text = L"Settings applied to " + v.cur.name + L". They take effect the next time the game starts."; v.toast.tone = 1;
        });
        // Ray Reconstruction (rr-notes.md): Cyberpunk 2077 uses it (the fixture's rr=installed); its section scrolled into
        // view (its first control at 40 % of the window), the Quality picks, Exact behind Settings' switch or set in the
        // game, off (unsaved), no RR data (Convert), converting, an install from before the RR files (Repair)
        auto rrView = [&] {
            RenderFrame((float)w, (float)h, s, L"");
            for (auto& x : v.hits)
                if (x.id == ID_G_RRON || x.id == ID_G_RRCONVERT || x.id == ID_G_RRREPAIR) { v.scroll[P_GAME] = v.target[P_GAME] = std::max(0.f, v.scroll[P_GAME] + x.r.cy() - v.H * 0.4f); break; }
        };
        auto rrSaved = [&](const char* key, const char* val) { SetSetting(v.edit, key, val, nullptr); v.saved = v.edit; };
        frame(L"game-rr", [&] { game(0); rrView(); });
        frame(L"game-rr-fast", [&] { game(0); rrSaved("rr_mode", "fast"); rrView(); });
        frame(L"game-rr-exact", [&] { game(0); gl.global.exactMode = true; rrView(); });
        frame(L"game-rr-exactsel", [&] { game(0); rrSaved("rr_mode", "exact"); rrView(); });  // switched to Exact in the game: shown without the switch
        frame(L"game-rr-off", [&] { game(0); SetSetting(v.edit, "rr_enabled", "off", nullptr); rrView(); });
        frame(L"game-rr-nodata", [&] { v.rrReady = false; game(0); rrView(); });
        frame(L"game-rr-converting", [&] { v.rrReady = false; game(0); v.busy = true; v.task = T_RRDATA; rrView(); });
        // the RR data missing while the game that uses it runs: Convert waits (disabled), and says why before a press
        frame(L"game-rr-nodata-running", [&] { v.rrReady = false; game(0); v.cur.status = GameStatus::Running; v.cur.statusText = L"Running (actions locked)"; rrView(); });
        frame(L"game-rr-notset", [&] { game(0); v.cur.rrInstalled = false; rrView(); });
        // its Set up running: the row says "Setting up..." with the busy indicator, the header the verb and the step
        frame(L"game-rr-settingup", [&] { game(0); v.cur.rrInstalled = false; v.busy = true; v.busyDir = v.cur.dir; v.busyVerb = L"Setting up\u2026"; v.step = L"Copying WrongGPU's files (2 of 4)"; v.progress = 0.3f; rrView(); });
        frame(L"game-rr-notused", [&] { game(0); v.cur.last.rrUsed = false; v.cur.last.rrMs.clear(); });  // Last run: a run without RR
        // r4 #1: Borderlands 4 after a run of Ray Reconstruction alone (no front-end log, as today's real runs): the page's top (the
        // header, no checklist), and its Last run (the Denoiser row, its cost and size)
        auto rrOnly = [&] {
            for (size_t i = 0; i < fixtures.size(); ++i) if (fixtures[i].e.name == L"Borderlands 4") { ShowFixtureGame(i); break; }
            v.cur.status = GameStatus::Installed; v.cur.installedVersion = Ctx().data->version; v.cur.statusText = L"Installed " + v.cur.installedVersion;
            v.cur.rrUses = v.cur.rrInstalled = true;
            SetSetting(v.saved, "rr_config_fix", "on", nullptr); v.edit = v.saved;
            v.rrcfg = 1; v.rrcfgFixed = true;
            v.rrcfgFile = L"C:\\Users\\Player\\Documents\\My Games\\Borderlands 4\\Saved\\Config\\Windows\\Engine.ini";
            LastRun& l = v.cur.last;
            l = LastRun();
            l.rrFound = l.rrUsed = true; l.rrFrames = 36044; l.rrMs = L"1.6"; l.rrMode = L"accurate"; l.renderSize = L"642x361"; l.outputSize = L"1920x1080";  // (rrUsed: the attempt, r6 #2)
            CountRrRun(l);
            l.when = L"2026-11-05 00:31";
            // installed at this version, so its RR kernels were prepared after the install (launcher_ui.cpp QueueRrPrep), and a game
            // run never changes that: the card says Prepared, as a real install does (r5 #8: the fixture had no prepared state)
            v.fixtureRrPrepared = v.cur.exe;
        };
        frame(L"game-rronly", [&] { rrOnly(); });
        frame(L"game-rronly-lastrun", [&] {
            rrOnly();
            RenderFrame((float)w, (float)h, s, L"");
            for (auto& x : v.hits)
                if (x.id == ID_G_LOG) { v.scroll[P_GAME] = v.target[P_GAME] = std::max(0.f, std::min(v.contentH[P_GAME] - v.H, x.full.y + x.scroll - v.H * 0.55f)); break; }
        });
        // r5 #3: Cyberpunk 2077 with Ray Reconstruction in one game session (both logs, the same process), from log lines through
        // the launcher's own parse: Model M's row and the header follow the front end's own evidence. Case A ("unused"): Model M
        // set, the game asked it for nothing; case B ("late"): the runtime ready, no frame of Model M's own logged
        auto rrMixed = [&](bool late) {
            game(0);
            v.cur.status = GameStatus::Installed; v.cur.adoptable = false; v.cur.problems.clear();
            v.cur.rrUses = v.cur.rrInstalled = true;
            const string fe = string("[20:00:01.000 1500] [x] ==== session start: Cyberpunk2077.exe (process 1500) ====\r\n"
                                     "[20:00:01.100 1500] [x] settings: upscaler=network\r\n") +
                              (late ? "[20:00:05.000 1500] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n" : "");
            const string rr = "20:00:00.000 load: process 1500 attached C:\\Games\\Cyberpunk 2077\\bin\\x64\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                              "20:00:00.100 hook: rr_mode accurate\r\n"
                              "20:19:59.000 runtime: instance 1 timing, 300 frames from frame 30601: GPU 3.579 ms per frame (pre 0.025), render 1280x720, output 1920x1080\r\n"
                              "20:20:00.500 runtime: destroy instance 1 after 30962 frames (0 failed)\r\n";
            LastRun l = ParseLastRun(fe);
            ParseRrLog(rr, l);
            CountRrRun(l);
            l.when = L"2026-11-05 20:20";
            l.optiPlugin = true;  // (the edition's slot: chosen by name, so no foot about it)
            v.cur.last = l;
        };
        frame(L"game-rrmixed-unused", [&] { rrMixed(false); });
        frame(L"game-rrmixed-late", [&] { rrMixed(true); });
        // r6 #2: an RR attempt without a successful frame (the lines of the synthetic tests/logs/rr-failed-only.log): a run that says
        // so, with the log's reason; no Worked, the checklist stays. Borderlands 4 alone (RR-only), and Cyberpunk 2077 in one session
        // with the front end where Model M wasn't used (the warning names Model M beside RR's evidence, r6 #1)
        const string rrFailedLog = "20:00:05.120 runtime: create instance 1: output 1920x1080, render 1280x720, mode 3\r\n"
                                   "20:00:05.130 runtime: instance 1 evaluate failed before any pass: frame kernels still being prepared (first start with a cold driver cache)\r\n"
                                   "20:00:05.131 rr: evaluate 1: runtime returned 13; no pass-through (11), the code goes to the game\r\n"
                                   "20:01:40.000 runtime: destroy instance 1 after 100 frames (100 failed)\r\n";
        auto rrFailed = [&] {
            rrOnly();
            LastRun& l = v.cur.last;
            l = LastRun();
            ParseRrLog("20:00:00.000 load: process 4100 attached C:\\Games\\Borderlands 4\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                       "20:00:00.100 hook: rr_mode accurate\r\n" + rrFailedLog, l);
            CountRrRun(l);
            l.when = L"2026-11-05 20:01";
        };
        frame(L"game-rrfailed", [&] { rrFailed(); });
        frame(L"game-rrfailed-lastrun", [&] {
            rrFailed();
            RenderFrame((float)w, (float)h, s, L"");
            for (auto& x : v.hits)
                if (x.id == ID_G_LOG) { v.scroll[P_GAME] = v.target[P_GAME] = std::max(0.f, std::min(v.contentH[P_GAME] - v.H, x.full.y + x.scroll - v.H * 0.55f)); break; }
        });
        frame(L"game-rrmixed-failed", [&] {
            rrMixed(false);
            LastRun l = ParseLastRun("[20:00:01.000 1500] [x] ==== session start: Cyberpunk2077.exe (process 1500) ====\r\n"
                                     "[20:00:01.100 1500] [x] settings: upscaler=network\r\n");
            ParseRrLog("20:00:00.000 load: process 1500 attached C:\\Games\\Cyberpunk 2077\\bin\\x64\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                       "20:00:00.100 hook: rr_mode accurate\r\n" + rrFailedLog, l);
            CountRrRun(l);
            l.when = L"2026-11-05 20:01";
            l.optiPlugin = true;
            v.cur.last = l;
        });
        // r7 #2: one session where Model M upscaled frames and the RR attempt finished none: Model M's run, the amber RR row, no green
        // "No warnings"
        frame(L"game-rrmixed-failed-ok", [&] {
            rrMixed(true);
            LastRun l = ParseLastRun("[20:00:01.000 1500] [x] ==== session start: Cyberpunk2077.exe (process 1500) ====\r\n"
                                     "[20:00:01.100 1500] [x] settings: upscaler=network\r\n"
                                     "[20:00:05.000 1500] [x]   U1 upscaler info: network runtime ready: executor dx12, sync inline\r\n"
                                     "[20:00:05.100 1500] [fe] #12 WrongGPU frame: R=1280x720 O=1920x1080\r\n[20:00:05.100 1500] [fe] #12 -> OK (WrongGPU)\r\n");
            ParseRrLog("20:00:00.000 load: process 1500 attached C:\\Games\\Cyberpunk 2077\\bin\\x64\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                       "20:00:00.100 hook: rr_mode accurate\r\n"
                       "20:00:05.120 runtime: create instance 1: output 1920x1080, render 1280x720, mode 3\r\n"
                       "20:00:05.130 runtime: instance 1 evaluate failed before any pass: runtime not ready\r\n"
                       "20:00:05.131 rr: evaluate 1: runtime returned 13; colour passed through, success reported, reset on the next evaluate\r\n"
                       "20:01:40.000 runtime: destroy instance 1 after 100 frames (100 failed)\r\n", l);
            CountRrRun(l);
            l.when = L"2026-11-05 20:01";
            l.optiPlugin = true;
            v.cur.last = l;
        });
        // r7 #3: the plugin's direct route refused the game's RR feature before any instance (tests/logs/rr-refused-only.log's lines)
        frame(L"game-rrrefused", [&] {
            rrOnly();
            LastRun& l = v.cur.last;
            l = LastRun();
            ParseRrLog("19:30:00.000 load: process 5200 attached C:\\Games\\Borderlands 4\\plugins\\WrongGPU-RR.asi (hook mode)\r\n"
                       "19:30:40.100 RR feature refused (the game keeps its own path): WrongGPU's runtime or model is not available\r\n", l);
            CountRrRun(l);
            l.when = L"2026-11-05 19:35";
        });
        // a game just installed, not played yet: "Turn it on in the game" leads the page (UI review #1); then its run where the
        // game asked for no upscaler (step 2 still open), and one where it did but Model M didn't start (step 2 done)
        auto fresh = [&](bool ran, bool asked) {
            game(0);
            LastRun& l = v.cur.last;
            l = LastRun();
            if (ran) { l.found = true; l.when = L"2026-11-04 21:10"; l.upscaler = L"network"; l.created = asked; }
            v.cur.rrInstalled = false;
        };
        frame(L"game-guide", [&] { fresh(false, false); });
        frame(L"game-guide-asked", [&] { fresh(true, true); });
        // r6 #4: before the first run with WrongGPU's Ray Reconstruction on (Cyberpunk 2077: RR set up and on): step 3 names the RR
        // form too
        frame(L"game-guide-rr", [&] { fresh(false, false); v.cur.rrUses = v.cur.rrInstalled = v.cur.rrEnabled = true; });
        // the runtime ready but no frame upscaled yet (it logs "ready" before any frame): the checklist stays (UI review r2 #10)
        frame(L"game-guide-readyonly", [&] { fresh(true, true); v.cur.last.ready = true; v.cur.last.executor = L"dx12"; });
        // step 3 follows the OptiScaler (r3 #3): WrongGPU's edition chosen by name (no foot), and another OptiScaler (AMD FSR)
        frame(L"game-guide-readyonly-edition", [&] { fresh(true, true); v.cur.last.ready = v.cur.last.optiPlugin = true; v.cur.last.executor = L"dx12"; });
        frame(L"game-guide-noedition", [&] { fresh(false, false); v.cur.optiEdition = false; });
        frame(L"game-guide-fps", [&] { fresh(false, false); SetSetting(v.edit, "fps_overlay", "on", nullptr); });
        // the FPS overlay saved on (it is on), and an OptiScaler without its key (no promise, no button) (UI review r2 #9)
        frame(L"game-guide-fps-on", [&] { fresh(false, false); SetSetting(v.edit, "fps_overlay", "on", nullptr); v.saved = v.edit; });
        frame(L"game-guide-nofps", [&] {
            fresh(false, false);
            for (size_t i = 0; i < v.opti.locked.size(); ++i)
                if (v.opti.locked[i].section == "Menu" && v.opti.locked[i].key == "ShowFps") { v.opti.locked.erase(v.opti.locked.begin() + (long long)i); break; }
        });
        // Borderlands 4's Engine.ini setting for Ray Reconstruction (UI review #2): the question after an install, and its page's
        // row: not chosen yet, WrongGPU (added), Myself with the steps and the file check (not there yet, there, not read-only)
        auto bl4 = [&](const char* fix, int state) {
            for (size_t i = 0; i < fixtures.size(); ++i) if (fixtures[i].e.name == L"Borderlands 4") { ShowFixtureGame(i); break; }
            // installed and up to date (the fixture's Borderlands 4 has 0.5.0 with an update ready: these frames show the installed
            // state, so this version, never "WrongGPU 0.5.0 is installed" without the update: r5 #9)
            v.cur.status = GameStatus::Installed; v.cur.rrUses = v.cur.rrInstalled = true;
            v.cur.installedVersion = Ctx().data->version; v.cur.statusText = L"Installed " + v.cur.installedVersion;
            v.fixtureRrPrepared = v.cur.exe;  // (installed at this version: its RR kernels were prepared after the install, r5 #8)
            SetSetting(v.saved, "rr_config_fix", fix, nullptr); v.edit = v.saved;
            v.rrcfg = state; v.rrcfgFile = L"C:\\Users\\Player\\Documents\\My Games\\Borderlands 4\\Saved\\Config\\Windows\\Engine.ini";
            v.rrcfgFixed = !strcmp(fix, "on");  // WrongGPU's choice applied (the record has its line); r2b #1's waiting frames clear it
            // the game's settings folder not there yet: it has never started, so no last run either (never "Worked last time you
            // played" beside "Waiting for the game's first start": r4 #12)
            if (state == -1) v.cur.last = LastRun();
            v.scroll[P_GAME] = v.target[P_GAME] = 330;
        };
        // the Engine.ini card scrolled under the pinned header (its segmented control 28 DIPs into it), its file check in view
        auto cardView = [&] {
            RenderFrame((float)w, (float)h, s, L"");
            for (auto& x : v.hits)
                if (x.id == ID_G_RRCFG0) {
                    v.scroll[P_GAME] = v.target[P_GAME] = std::max(0.f, std::min(v.contentH[P_GAME] - v.H, std::round(x.full.y + x.scroll - 28 - (kBarH + 60) - 16)));
                    break;
                }
        };
        // installed with an OptiScaler that isn't WrongGPU's edition: Ray Reconstruction can't run, the Engine.ini card stays (r2 #2)
        frame(L"game-rrconfig-noedition", [&] { bl4("ask", 0); v.cur.rrDirect = true; v.cur.optiEdition = false; cardView(); });
        // the file check in view, one frame per state, with either choice (r2 #6, #7): there, missing, not read-only, unknown
        static const struct { const wchar_t* tag; int state; } kRrChecks[] = {{L"there", 1}, {L"missing", 0}, {L"notro", 2}, {L"unknown", -1}};
        for (auto& c : kRrChecks)
            for (int mine = 0; mine < 2; ++mine) {
                const wstring nm = wstring(L"game-rrconfig-check-") + c.tag + (mine ? L"-myself" : L"-wronggpu");
                frame(nm.c_str(), [&] { bl4(mine ? "off" : "on", c.state); cardView(); });
            }
        // the player's own line, not read-only, WrongGPU's never added (no record line): Fix it now, not "again" (r3 #10)
        frame(L"game-rrconfig-check-notro-own", [&] { bl4("on", 2); v.rrcfgFixed = false; cardView(); });
        // an Engine.ini it can't read (saved as UTF-16): said as such, with what lets it be read; Check again (r3 #8)
        for (int mine = 0; mine < 2; ++mine) {
            const wstring nm = wstring(L"game-rrconfig-check-cantread") + (mine ? L"-myself" : L"-wronggpu");
            frame(nm.c_str(), [&] { bl4(mine ? "off" : "on", 3); v.rrcfgWhy = L"it is saved as UTF-16, which WrongGPU doesn't read"; cardView(); });
        }
        // Fix it for me before the game's first start (r2b #1): nothing written, the card says it waits for the game's settings
        // folder; the choice made on the page while it isn't there; the question's result toast
        frame(L"game-rrconfig-pending", [&] { bl4("on", -1); v.rrcfgFixed = false; cardView(); });
        frame(L"game-rrconfig-pending-choose", [&] { bl4("ask", -1); SetSetting(v.edit, "rr_config_fix", "on", nullptr); cardView(); });
        frame(L"game-rrconfig-pending-toast", [&] {
            bl4("on", -1); v.rrcfgFixed = false; cardView();
            OpResult res; res.notes.push_back(v.cur.name + L": " + kRrConfigWaitNote);
            ShowToast(OpSuccess(Op::Apply, res, OpRequest(), nullptr, v.cur.name), 1);
        });
        // the game in a protected folder (r2c): the window never adds the line by itself; waiting, then "Ready to add" with Fix it
        // now once the game's settings folder is there, and the question's result toast
        frame(L"game-rrconfig-pending-protected", [&] { bl4("on", -1); v.rrcfgFixed = false; v.cur.rrConfigProtected = true; cardView(); });
        // ...WrongGPU chosen on the page (not applied yet) before the game's first start: a click later, never by itself (r3 #4)
        frame(L"game-rrconfig-pending-choose-protected", [&] {
            bl4("ask", -1); v.rrcfgFixed = false; v.cur.rrConfigProtected = true; SetSetting(v.edit, "rr_config_fix", "on", nullptr); cardView();
        });
        frame(L"game-rrconfig-ready", [&] { bl4("on", 0); v.rrcfgFixed = false; v.cur.rrConfigProtected = true; cardView(); });
        // r7 #5: an install without WrongGPU's RR files: the card says so; the line waits for the player's click (never added by the
        // window itself), with no permission sentence in a folder that needs none: before and after the game's first start
        frame(L"game-rrconfig-norr", [&] { bl4("on", 0); v.rrcfgFixed = false; v.cur.rrInstalled = false; cardView(); });
        frame(L"game-rrconfig-norr-pending", [&] { bl4("on", -1); v.rrcfgFixed = false; v.cur.rrInstalled = false; cardView(); });
        frame(L"game-rrconfig-pending-toast-protected", [&] {
            bl4("on", -1); v.rrcfgFixed = false; v.cur.rrConfigProtected = true; cardView();
            OpResult res; res.notes.push_back(v.cur.name + L": " + kRrConfigWaitNote);
            ShowToast(OpSuccess(Op::Apply, res, OpRequest(), &v.cur, v.cur.name), 1);
        });
        frame(L"dialog-rrconfig", [&] {
            bl4("ask", 0); v.scroll[P_GAME] = v.target[P_GAME] = 0;
            v.dialog.open = true; v.dialog.icon = 0;
            if (const RrConfigGame* rg = RrConfigGameFor(L"Borderlands4.exe")) SplitDialogText(RrConfigNotice(*rg), v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Not now", 0}, {L"Show me how", 2}, {L"Fix it for me", 1, true}};
            v.focus = ID_DIALOG + 2; v.focusVisible = false;
        });
        // ...for a game in a protected folder: Fix it for me brings Windows' permission prompt, and the offer says so (r4 #9)
        frame(L"dialog-rrconfig-protected", [&] {
            bl4("ask", 0); v.scroll[P_GAME] = v.target[P_GAME] = 0; v.cur.rrConfigProtected = true;
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(RrConfigQuestionText(v.cur.dir, v.cur.exe), v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Not now", 0}, {L"Show me how", 2}, {L"Fix it for me", 1, true}};
            v.focus = ID_DIALOG + 2; v.focusVisible = false;
        });
        // ...and before the game's first start there: the prompt comes later, at Fix it now on its page after that start (r5 #9)
        frame(L"dialog-rrconfig-protected-firststart", [&] {
            bl4("ask", -1); v.scroll[P_GAME] = v.target[P_GAME] = 0; v.cur.rrConfigProtected = true;
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(RrConfigQuestionText(v.cur.dir, v.cur.exe), v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Not now", 0}, {L"Show me how", 2}, {L"Fix it for me", 1, true}};
            v.focus = ID_DIALOG + 2; v.focusVisible = false;
        });
        frame(L"game-rrconfig-ask", [&] { bl4("ask", 0); });
        frame(L"game-rrconfig-on", [&] { bl4("on", 1); });
        frame(L"game-rrconfig-myself", [&] { bl4("off", 0); });
        frame(L"game-rrconfig-myself-found", [&] { bl4("off", 1); });
        frame(L"game-rrconfig-myself-notro", [&] { bl4("off", 2); });
        // "Show me how" (UI review r2 #1): the page opened at its top, then scrolled to the Engine.ini card once its steps show,
        // the keyboard on Open folder (as RevealRrConfigCard does after the Apply)
        frame(L"game-rrconfig-showme", [&] {
            bl4("off", 0); v.scroll[P_GAME] = v.target[P_GAME] = 0;
            ShowRrConfigSteps(v.cur.dir);
            RenderFrame((float)w, (float)h, s, L"");
        });
        // ...answered for a game in a protected folder (r3 #6): the steps at once, Myself unsaved (Apply saves it), no prompt
        frame(L"game-rrconfig-showme-protected", [&] {
            bl4("ask", 0); v.cur.rrConfigProtected = true; v.scroll[P_GAME] = v.target[P_GAME] = 0;
            for (auto& e : v.games) if (IEq(e.dir, v.cur.dir)) e.rrConfigProtected = true;
            AnswerRrConfigQuestion(v.cur.dir, v.cur.exe, v.saved, 2);
            RenderFrame((float)w, (float)h, s, L"");
        });
        // Last run: a run where the game never asked for an upscaler (TAA in its settings): WrongGPU wasn't used (UI review #9)
        frame(L"game-notused", [&] {
            game(0);
            LastRun& l = v.cur.last;
            l.ready = l.created = l.upscaled = l.rrUsed = false; l.executor.clear(); l.renderSize.clear(); l.outputSize.clear(); l.rrMs.clear(); l.notes.clear();
        });
        // the same run on the direct route (no OptiScaler): the game's AMD FSR is what reaches WrongGPU (UI review r2 #3)
        frame(L"game-notused-direct", [&] {
            game(0);
            v.cur.hosted = false;
            LastRun& l = v.cur.last;
            l.ready = l.created = l.upscaled = l.rrUsed = false; l.executor.clear(); l.renderSize.clear(); l.outputSize.clear(); l.rrMs.clear(); l.notes.clear();
        });
        // the Upscaler's Use WrongGPU row in view (under the pinned header), on OptiScaler's route and on the direct one: what
        // turning it off does there (r2b #3)
        auto upscalerView = [&] {
            RenderFrame((float)w, (float)h, s, L"");
            for (auto& x : v.hits)
                if (x.id == ID_G_MASTER) {
                    v.scroll[P_GAME] = v.target[P_GAME] = std::max(0.f, std::min(v.contentH[P_GAME] - v.H, std::round(x.full.y + x.scroll - 90 - (kBarH + 60) - 16)));
                    break;
                }
        };
        frame(L"game-upscaler", [&] { game(0); upscalerView(); });
        frame(L"game-upscaler-direct", [&] { game(0); v.cur.hosted = false; upscalerView(); });
        // RE Requiem (gamerules.h): it needs the player's REFramework, and its RR comes only with path tracing. Built here on
        // the first fixture game's settings, so the Library's frames stay as they are. state: 0 ready to install (REFramework
        // found), 1 needs REFramework (Install held back, Learn more), 2 installed, 3 installed and REFramework gone
        static const wchar_t* const kRe9Dir = L"M:\\SteamLibrary\\steamapps\\common\\RESIDENT EVIL requiem BIOHAZARD requiem";
        auto re9 = [&](int state) {
            game(0);
            GameEntry e;
            e.name = L"RESIDENT EVIL requiem";
            e.appid = L"3764200";
            e.source = L"Steam";
            e.dir = kRe9Dir;
            e.exe = Join(e.dir, L"re9.exe");
            e.exeFound = true;
            e.rrUses = true; e.rrWhy = L"sl.dlss_d.dll"; e.rrPathTracingOnly = true;  // DLSS Ray Reconstruction's plugin
            e.needs = L"REFramework"; e.needsPage = L"https://github.com/praydog/REFramework"; e.needsGame = L"RE Requiem";
            e.needsFound = state != 1 && state != 3;
            if (e.needsFound) {
                e.needsWhy = L"dinput8.dll, folder reframework";
                Finding f; f.tool = Tool::ReFramework; f.decision = Decision::Inform; f.what = L"REFramework as dinput8.dll"; f.files = {L"dinput8.dll", L"reframework"};
                f.evidence = L"text \"REFramework\""; f.message = L"Stays where it is: RE Requiem needs it before OptiScaler can run.";
                e.coexist.push_back(f);
            }
            e.last = LastRun();
            if (state >= 2) {
                e.hasRecord = e.hosted = true; e.installedVersion = Ctx().data->version; e.proxy = L"dxgi.dll";
                e.status = GameStatus::Installed; e.statusText = L"Installed " + e.installedVersion;
                if (state == 3) {  // REFramework gone since the install: the core's status says what the game needs (launcher.cpp)
                    e.status = GameStatus::NeedsPrereq;
                    e.statusText = L"Installed " + e.installedVersion + L", needs REFramework: RE Requiem blocks OptiScaler without it";
                }
                e.rrInstalled = true; e.rrEnabled = true; e.rrMode = L"accurate";
                e.last.found = true; e.last.when = L"2026-11-02 05:18"; e.last.upscaler = L"network"; e.last.executor = L"dx12"; e.last.ready = e.last.upscaled = true;
                e.last.renderSize = L"1706x960"; e.last.outputSize = L"2560x1440";
                e.last.rrFound = e.last.rrUsed = true; e.last.rrMode = L"accurate"; e.last.rrMs = L"7.1"; e.last.rrFrames = 1;  // (its frames: r6 #2)
                e.last.optiPlugin = true;
                v.opti.folder = e.dir; v.opti.proxy = e.proxy;
            } else {
                e.status = state == 1 ? GameStatus::NeedsPrereq : GameStatus::Eligible;
                e.statusText = state == 1 ? L"Needs REFramework: RE Requiem blocks OptiScaler without it" : L"Ready to install (experimental)";
                e.proxyPlan.name = L"dxgi.dll"; e.proxyPlan.reason = L"the first free name";
                v.opti = OptiView();
                v.saved = DefaultProfile();
            }
            ShowEntry(e, false);
        };
        frame(L"game-re9", [&] { re9(2); });
        frame(L"game-re9-rr", [&] { re9(2); rrView(); });
        frame(L"game-re9-rr-off", [&] { re9(2); SetSetting(v.edit, "rr_enabled", "off", nullptr); rrView(); });
        frame(L"game-re9-ready", [&] { re9(0); });
        frame(L"game-re9-needs", [&] { re9(1); });
        frame(L"game-re9-needs-focus", [&] { re9(1); v.focus = ID_G_LEARN; v.focusVisible = true; });
        frame(L"game-re9-gone", [&] { re9(3); });
        // a build without WrongGPU's OptiScaler edition: RE Requiem's RR can't run, and the page says so (no switch, no "on")
        frame(L"game-re9-noedition", [&] {
            re9(2); v.rrEdition = false;
            RenderFrame((float)w, (float)h, s, L"");
            for (auto& x : v.hits)  // its sentence in view, as rrView brings the RR rows
                if (x.name.rfind(L"This version of WrongGPU", 0) == 0) { v.scroll[P_GAME] = v.target[P_GAME] = std::max(0.f, x.r.cy() - v.H * 0.4f); break; }
        });
        frame(L"library-re9", [&] {  // the game that needs REFramework on the Library's Available shelf (amber line, Details)
            GameEntry e;
            e.name = L"RESIDENT EVIL requiem"; e.appid = L"3764200"; e.source = L"Steam";
            e.dir = kRe9Dir; e.exe = Join(e.dir, L"re9.exe"); e.exeFound = true;
            e.needs = L"REFramework"; e.needsGame = L"RE Requiem"; e.status = GameStatus::NeedsPrereq;
            e.statusText = L"Needs REFramework: RE Requiem blocks OptiScaler without it";
            v.games.push_back(e);
            v.search = L"requiem";
        });
        frame(L"library-re9-gone", [&] {  // installed, REFramework removed since: on "Needs attention", amber, Details
            GameEntry e;
            e.name = L"RESIDENT EVIL requiem"; e.appid = L"3764200"; e.source = L"Steam";
            e.dir = kRe9Dir; e.exe = Join(e.dir, L"re9.exe"); e.exeFound = true;
            e.hasRecord = e.hosted = true; e.installedVersion = Ctx().data->version;
            e.needs = L"REFramework"; e.needsGame = L"RE Requiem"; e.status = GameStatus::NeedsPrereq;
            e.statusText = L"Installed " + e.installedVersion + L", needs REFramework: RE Requiem blocks OptiScaler without it";
            v.games.push_back(e);
        });
        frame(L"game-update", [&] { game(gUpdate); });
        frame(L"game-busy", [&] { game(gUpdate); v.busy = true; v.busyDir = v.cur.dir; v.busyVerb = L"Updating…"; v.step = L"Copying WrongGPU's files (3 of 7)"; v.progress = 0.4f; });
        frame(L"game-repair", [&] { game(gRepair); });
        // two of WrongGPU's files missing or changed: one row with their count, the names in its tooltip
        frame(L"game-repair-files", [&] { game(gRepair); v.cur.problems = {L"missing: dxgi.dll", L"changed: bin\\x64\\plugins\\wronggpu_rr.asi"}; });
        // Install running on a game page: its button says "Installing…", the line the verb and then the step
        frame(L"game-installing", [&] { game(gEligible); v.busy = true; v.busyDir = v.cur.dir; v.busyVerb = L"Installing…"; v.step = L"Backing up the game's files"; v.progress = 0.2f; });
        // a game added with Add game: Remove from Library in its More menu
        frame(L"game-added-menu", [&] { game(gEligible); v.cur.source = L"Added"; OpenMoreMenu(); });
        frame(L"game-noart", [&] { game(gNoArt); });
        frame(L"game-refused", [&] { game(gRefused); });
        frame(L"game-longname", [&] { game(gLong); });
        frame(L"game-notice", [&] {  // a failed operation's callout on its game's page (the note was shown as it failed)
            game(0);
            v.notice = OpFailLead(Op::Repair) + L" " + PlainCause(L"Cannot rename C:\\Games\\Control\\dxgi.dll: The process cannot access the file because it is being used by another "
                                                              L"process. Is the game running?");
            v.noticeTone = 2; v.noticeDir = v.cur.dir;
        });
        // the other states of a game's page (the fixture's records with another status: nothing here is read from a folder)
        auto as = [&](int i, GameStatus st, const wstring& text) { game(i); v.cur.status = st; v.cur.statusText = text; };
        auto leftovers = [] { v.cur.leftovers = {{L"dxgi.dll", L"", L"0.5.0"}, {L"wronggpu_runtime.dll", L"", L"0.5.0"}, {L"wronggpu.ini", L"", L"0.5.0"}}; };
        frame(L"game-running", [&] { as(0, GameStatus::Running, L"Running (actions locked)"); });
        // the game running with More OptiScaler settings open: the number fields locked with their rows (dimmed, no keys)
        frame(L"game-running-more", [&] { as(0, GameStatus::Running, L"Running (actions locked)"); v.optiMore = true; v.scroll[P_GAME] = v.target[P_GAME] = 300; });
        // a game with anti-cheat, as the core reports it: the Library's sentence, the marker's path only in the tooltip
        frame(L"game-anticheat", [&] {
            const wstring ac = L"K:\\SteamLibrary\\steamapps\\common\\Dead Harbor\\EasyAntiCheat\\EasyAntiCheat_EOS_Setup.exe";
            as(gRefused >= 0 ? gRefused : 0, GameStatus::Refused, L"Refused: anti-cheat (" + ShortenMiddle(ac, 60) + L")");
            v.cur.coexist.clear();
            Finding f; f.tool = Tool::AntiCheat; f.decision = Decision::Refuse; f.what = L"Anti-cheat (" + ShortenMiddle(ac, 60) + L")"; f.evidence = ac;
            f.message = L"Found: " + ac + L". WrongGPU is never installed into games with anti-cheat; only removing WrongGPU's own recorded install is allowed.";
            v.cur.coexist.push_back(f);
        });
        // a Contrast theme (Windows' Accessibility > Contrast themes): the system's colours, an edge round every control
        frame(L"contrast-library", [&] { contrast(); v.hot = ID_TILEQA + 1; });
        frame(L"contrast-game", [&] { contrast(); game(0); SetSetting(v.edit, "sparkle_guard", "on", nullptr); v.focus = ID_G_PLAY; v.focusVisible = true; });
        frame(L"contrast-settings", [&] { contrast(); v.page = P_SETTINGS; });
        frame(L"contrast-dialog", [&] {
            contrast(); game(0);
            SetSetting(v.edit, "sparkle_guard", "off", nullptr);
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(PlayUnsavedText(v.cur.name), v.dialog.title, v.dialog.body);
            v.dialog.buttons = PlayUnsavedButtons();
        });
        frame(L"dialog-play-unsaved", [&] {  // Play with unsaved changes: Apply and play, Play without them, Cancel
            game(0);
            SetSetting(v.edit, "sparkle_guard", "off", nullptr);
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(PlayUnsavedText(v.cur.name), v.dialog.title, v.dialog.body);
            v.dialog.buttons = PlayUnsavedButtons();
            v.focus = ID_DIALOG + 2; v.focusVisible = false;
        });
        frame(L"game-adopt", [&] { game(0); v.cur.adoptable = true; });  // set up outside the launcher: Adopt beside Play
        frame(L"game-turnedoff", [&] {
            as(0, GameStatus::TurnedOff, L"Installed " + v.cur.installedVersion + L" (turned off)");
            SetSetting(v.saved, "wronggpu", "off", nullptr); v.edit = v.saved;
        });
        frame(L"game-orphaned", [&] { as(gUpdate, GameStatus::Orphaned, L"Orphaned: the game is gone"); });
        frame(L"game-oldinstall", [&] { as(gEligible, GameStatus::OldInstall, L"An older WrongGPU install (0.4.5)"); });
        frame(L"game-recordv1", [&] { game(gUpdate); v.cur.recordV1 = true; v.scroll[P_GAME] = v.target[P_GAME] = 5000; });
        frame(L"game-leftovers", [&] { as(gEligible, GameStatus::Leftovers, L"Leftover WrongGPU files without a record"); leftovers(); });
        frame(L"dialog-leftovers", [&] {  // Clean up's question: the files in words, their names behind Show details
            as(gEligible, GameStatus::Leftovers, L"Leftover WrongGPU files without a record"); leftovers();
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(LeftoversText(v.cur.name, v.cur.leftovers, true), v.dialog.title, v.dialog.body);
            v.dialog.buttons = LeftoversButtons(true);
            v.focus = ID_DIALOG + 1; v.focusVisible = false;  // destructive: the keyboard on Cancel (RunDialog)
        });
        frame(L"game-yours", [&] {  // the player's own OptiScaler
            as(gEligible, GameStatus::PlayersOptiScaler, L"Has your own OptiScaler: Install asks to switch to WrongGPU's OptiScaler setup");
            // as the planner has it: the player's OptiScaler as winmm.dll, ReShade as dxgi.dll, WrongGPU's setup where theirs was
            Finding f; f.tool = Tool::OptiScaler; f.decision = Decision::AskSwitch; f.what = L"Your OptiScaler 0.7.7 as winmm.dll";
            f.files = {L"winmm.dll", L"OptiScaler.ini"};
            f.message = L"Install asks you to switch to WrongGPU's OptiScaler setup: your OptiScaler files are backed up and put back byte for byte when WrongGPU is "
                        L"uninstalled. (Version 0.7.7: untested with WrongGPU.)";
            v.cur.coexist.push_back(f);
            Finding c; c.tool = Tool::ReShade; c.decision = Decision::Chain; c.what = L"ReShade 6.3.3 as dxgi.dll"; c.files = {L"dxgi.dll", L"ReShade.ini"};
            c.message = L"Stays where it is; OptiScaler takes another free name (an untested combination). Its hooks give command lists their own tables, "
                        L"which can make WrongGPU fall back to AMD FSR: the game page shows it after a run.";
            v.cur.coexist.push_back(c);
            v.cur.proxyPlan.name = L"winmm.dll"; v.cur.proxyPlan.reason = L"where your OptiScaler was";
        });
        frame(L"game-foreign", [&] {
            as(gEligible, GameStatus::ForeignMod, L"Other upscaler mod: not touched");
            Finding f; f.tool = Tool::NrOnAmd; f.decision = Decision::Refuse; f.what = L"Another upscaler mod"; f.evidence = L"its own settings or log file";
            f.message = L"WrongGPU does not install next to it and does not touch it.";
            v.cur.coexist.push_back(f);
            // and another HIP tool (BeamNG.drive's case): its warning is moot where WrongGPU stays out (no Runs on to name)
            Finding h; h.tool = Tool::HipRuntime; h.decision = Decision::Warn; h.what = L"A HIP-based tool (neuralhub.ini)"; h.files = {L"neuralhub.ini"};
            h.message = L"Its own HIP runtime may be loaded into the game next to WrongGPU's: keep WrongGPU on the D3D12 executor (the default). version.dll is never taken as OptiScaler's name here.";
            v.cur.coexist.push_back(h);
        });
        frame(L"game-outside", [&] {  // values changed outside the launcher, in wronggpu.ini and in OptiScaler.ini: Keep / Put back
            game(0);
            v.outside = {{"sparkle_guard", "on", "off"}, {"executor", "hip", "dx12"}};
            v.opti.outside = {{"Menu", "ShortcutKey", "0x70", "0x2D"}};
            v.scroll[P_GAME] = v.target[P_GAME] = 300;
        });
        frame(L"game-outside-bottom", [&] {
            game(0);
            v.outside = {{"sparkle_guard", "on", "off"}};
            v.opti.outside = {{"Menu", "ShortcutKey", "0x70", "0x2D"}};
            v.scroll[P_GAME] = v.target[P_GAME] = 5000;
        });
        frame(L"game-details", [&] { game(0); v.detailsOpen = true; v.scroll[P_GAME] = v.target[P_GAME] = 200; });  // the log's lines open
        frame(L"game-pinned-long", [&] {  // the longest name and three actions in the compact header
            as(gLong, GameStatus::Leftovers, L"Leftover WrongGPU files without a record"); leftovers();
            v.scroll[P_GAME] = v.target[P_GAME] = 560;
        });
        // Settings
        frame(L"settings", [] { v.page = P_SETTINGS; });
        // the accent row in every accent, scrolled to the Launcher section
        for (int a = 0; a < A_COUNT; ++a)
            frame((L"settings-accent-" + wstring(AccentKey(a))).c_str(), [&, a] {
                accent(a);
                v.page = P_SETTINGS;
                RenderFrame((float)w, (float)h, s, L"");
                v.scroll[P_SETTINGS] = v.target[P_SETTINGS] = SettingsSectionScroll(2);
            });
        frame(L"settings-options", [&] { v.page = P_SETTINGS; RenderFrame((float)w, (float)h, s, L""); Activate(ID_S_MODELMORE); });  // a ready model's Options... menu
        frame(L"settings-rr-missing", [] { v.page = P_SETTINGS; v.rrReady = false; });  // Ray Reconstruction data not converted
        frame(L"settings-rr-exact", [] { v.page = P_SETTINGS; gl.global.exactMode = true; });
        frame(L"settings-rr-options", [&] { v.page = P_SETTINGS; RenderFrame((float)w, (float)h, s, L""); Activate(ID_S_RRMORE); });
        frame(L"settings-rr-converting", [] { v.page = P_SETTINGS; v.busy = true; v.task = T_RRDATA; });
        // a game with WrongGPU's RR files running: the data rows' buttons wait for it (disabled, the reason in the tooltip and
        // help text): the model's Options... and the RR data's Choose a DLSS file... and Convert
        frame(L"settings-running", [] {
            v.page = P_SETTINGS; v.rrReady = false;
            for (auto& g0 : v.games) if (g0.rrInstalled && g0.hasRecord) { g0.status = GameStatus::Running; g0.statusText = L"Running (actions locked)"; break; }
        });
        // no game with WrongGPU's RR files: the RR data is optional (a grey dot, a plain Convert, the sentence says who needs it)
        frame(L"settings-rr-optional", [] { v.page = P_SETTINGS; v.rrReady = false; for (auto& g0 : v.games) g0.rrInstalled = false; });
        // an operation in a game: the data rows' buttons wait, the reason in their tooltip and help text (BusyWhy)
        frame(L"settings-busy", [] {
            v.page = P_SETTINGS; v.rrReady = false;
            for (auto& g0 : v.games) if (g0.status == GameStatus::UpdateAvailable) { v.busyDir = g0.dir; break; }
            v.busy = true; v.busyVerb = L"Updating…"; v.step = L"Copying WrongGPU's files (3 of 7)"; v.progress = 0.3f;
        });
        frame(L"settings-bottom", [] { v.page = P_SETTINGS; v.scroll[P_SETTINGS] = v.target[P_SETTINGS] = 5000; });
        frame(L"settings-update", [] {  // scrolled so the release address and its buttons are in view
            v.page = P_SETTINGS; gl.global.updateCheck = true; v.urlText = L"https://github.com/example/wronggpu/releases";
            v.scroll[P_SETTINGS] = v.target[P_SETTINGS] = 5000;
            v.toast.text = L"Release address saved."; v.toast.tone = 1;
        });
        gl.global.updateCheck = false; v.urlText.clear();
        frame(L"settings-set", [] {  // everything set: two game folders, an OptiScaler download, the HIP runtime found
            v.page = P_SETTINGS;
            gl.global.extraFolders = {L"K:\\Games\\Itch", L"H:\\Emulated\\PC Ports"};
            gl.global.optiPackage = L"M:\\Downloads\\OptiScaler_0.9.0.7z";
            v.fixtureHip = L"C:\\Windows\\System32\\amdhip64_7.dll";  // what an install records (hip-runtime.txt)
        });
        gl.global.extraFolders.clear(); gl.global.optiPackage.clear(); v.fixtureHip.clear();
        frame(L"settings-installed", [] {  // a portable copy where the launcher is installed for this user: the state and the sentence agree (no Uninstall here)
            v.page = P_SETTINGS; v.launcherInstalled = true; v.scroll[P_SETTINGS] = v.target[P_SETTINGS] = 5000;
        });
        v.launcherInstalled = false;
        frame(L"settings-problems", [] {  // the amber rows: no model data, started inside an app package (the HIP runtime not found)
            v.page = P_SETTINGS; v.modelReady = false; v.fixturePackaged = true;
            v.scroll[P_SETTINGS] = v.target[P_SETTINGS] = 0;
        });
        frame(L"settings-problems-bottom", [] { v.page = P_SETTINGS; v.modelReady = false; v.fixturePackaged = true; v.scroll[P_SETTINGS] = v.target[P_SETTINGS] = 5000; });
        v.fixturePackaged = false; v.modelReady = true;
        // the first run, dialogs
        frame(L"first-run", [] { v.first.open = true; v.first.busy = false; v.first.installOffer = true; v.first.choice = 1; v.first.lines = fixtureFirstRun; });
        frame(L"first-run-portable", [] { v.first.open = true; v.first.busy = false; v.first.installOffer = true; v.first.choice = 2; v.first.lines = fixtureFirstRun; });
        frame(L"first-run-busy", [] { v.first.open = true; v.first.busy = true; v.first.installOffer = true; v.first.choice = 1; });
        frame(L"first-run-problem", [] {  // no usable model file: an amber line, the summary says so, the button that fixes it
            v.first.open = true; v.first.busy = false; v.first.installOffer = true; v.first.choice = 1;
            v.first.lines = fixtureFirstRun;
            for (auto& l : v.first.lines)
                if (l.second.rfind(L"Model M", 0) == 0) l = {2, L"Model M: Needs a recent game's DLSS file (version 310.5\u00a0or\u00a0newer)"};
        });
        frame(L"first-run-notes", [] {  // the live run's note: no HIP runtime (grey, behind Show details with the passed checks)
            v.first.open = true; v.first.busy = false; v.first.installOffer = true; v.first.choice = 1;
            v.first.lines = fixtureFirstRun;
            for (auto& l : v.first.lines)
                if (l.second.rfind(L"HIP runtime", 0) == 0) l = {1, L"HIP runtime: Not found (optional). Only Sparkle guard and the HIP choice in Runs\u00a0on need it."};
        });
        frame(L"first-run-partial", [] {  // the checks arriving: the graphics card's done, the model data and the games still running
            v.first.open = true; v.first.busy = true; v.first.installOffer = true; v.first.choice = 1;
            for (auto& l : fixtureFirstRun) { if (l.second.rfind(L"Model M", 0) == 0) break; v.first.lines.push_back(l); }
        });
        frame(L"first-run-checks", [] {  // every check shown (Show details); in a short window the part above the foot scrolls
            v.first.open = true; v.first.busy = false; v.first.installOffer = true; v.first.choice = 2; v.first.lines = fixtureFirstRun; v.first.checksOpen = true;
        });
        frame(L"first-run-checks-focus", [] {  // Hide details with the keyboard focus: its ring clear of the line above it
            v.first.open = true; v.first.busy = false; v.first.installOffer = true; v.first.choice = 2; v.first.lines = fixtureFirstRun; v.first.checksOpen = true;
            v.focus = ID_FR_CHECKS; v.focusVisible = true;
        });
        frame(L"first-run-installed", [] {  // the launcher is installed already: no install offer, one step
            v.first.open = true; v.first.busy = false; v.first.installOffer = false; v.first.lines = fixtureFirstRun;
        });
        frame(L"dialog", [&] {  // Ask-to-switch OptiScaler over the game's page, as Install shows it
            game(gNoArt >= 0 ? gNoArt : 0);
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(L"Switch to WrongGPU's OptiScaler setup?\n\nThis game already has OptiScaler 0.7.7 (yours). Your files are backed up and put back exactly as "
                            L"they were if you uninstall WrongGPU.", v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Cancel", 0}, {L"Switch", 1, true}};
            v.focus = ID_DIALOG + 1; v.focusVisible = false;  // the default: the action the player chose (RunDialog); its fill shows it, no ring
        });
        frame(L"dialog-uninstall", [&] {
            game(0);
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(UninstallGameText(v.cur.name, false, 0), v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Cancel", 0}, {L"Uninstall", 1, true, true}};
        });
        frame(L"dialog-uninstall-gone", [&] {  // a game that is gone (Orphaned): Remove files, the last game with WrongGPU
            as(gUpdate, GameStatus::Orphaned, L"Orphaned: the game is gone");
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(UninstallGameText(v.cur.name, true, 1), v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Cancel", 0}, {L"Remove files", 1, true, true}};
        });
        // Settings > Uninstall launcher... (and Apps & features): a question with the mark and no default; a game has WrongGPU (Collect logs
        // offered), no game left after "from games too", so the model and RR data go too
        frame(L"dialog-uninstall-launcher", [&] {
            v.page = P_SETTINGS;
            v.dialog.open = true; v.dialog.icon = 0;
            LauncherUninstallLook look; look.anyGame = true; look.rrData = true;
            SplitDialogText(UninstallLauncherText(look, true, true), v.dialog.title, v.dialog.body);
            v.dialog.buttons = UninstallLauncherButtons(true, true);
            v.focus = 0; v.focusVisible = false;
        });
        frame(L"dialog-uninstall-launcher-alone", [&] {  // no game has WrongGPU: one plain question, Cancel or Uninstall
            v.page = P_SETTINGS;
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(UninstallLauncherText(LauncherUninstallLook(), true, false), v.dialog.title, v.dialog.body);
            v.dialog.buttons = UninstallLauncherButtons(false, false);
            v.focus = 0; v.focusVisible = false;
        });
        frame(L"dialog-remove-data", [&] {  // a portable copy's Settings > Remove data...: the shared data before its folder goes
            v.page = P_SETTINGS; v.scroll[P_SETTINGS] = v.target[P_SETTINGS] = 5000;
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(RemoveDataText(true, true), v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Cancel", 0, false, false}, {L"Remove data", 1, true, true}};  // as Confirm(..., danger) builds them
            v.focus = 0; v.focusVisible = false;
        });
        frame(L"dialog-update", [&] {  // a newer release (the opt-in check): Not now, or its page in the browser
            v.page = P_LIBRARY;
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(L"WrongGPU 0.5.6 is available.\n\nYou have WrongGPU " + wstring(g.ctx.data->version) + L". The release page has the download and what changed.",
                            v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Not now", 0}, {L"Open release page", 1, true}};
            v.focus = ID_DIALOG + 1; v.focusVisible = false;
        });
        frame(L"dialog-failure", [&] {  // a failed operation (WM_APP_OPDONE): the action and the game, the cause in words, the core's text behind Show details
            game(0);
            v.dialog.open = true; v.dialog.icon = kIconError;  // an error: the red glyph, Close the default
            SplitDialogText(OpFailTitle(Op::Repair, v.cur.name) + L"\n\n" + PlainCause(L"Cannot rename C:\\Games\\Control\\dxgi.dll: The process cannot access the file because "
                                                                                       L"it is being used by another process. Is the game running?"),
                            v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Show details", 2}, {L"Close", 1, true}};
            v.focus = ID_DIALOG + 1; v.focusVisible = false;
        });
        frame(L"dialog-long", [&] {  // a failure's details (its core text and warnings), longer than a short window: the text scrolls (the thumb, a fade)
            game(0);
            v.dialog.open = true; v.dialog.icon = 2;
            wstring t = L"Details\n\nCannot rename C:\\Games\\" + v.cur.name + L"\\bin\\x64\\dxgi.dll: The process cannot access the file because it is being used by another process. Is the game running?";
            for (int k = 1; k <= 8; ++k)  // taller than a 560 or 640 high window: there it scrolls
                t += L"\n\nWarning " + std::to_wstring(k) + L": a file WrongGPU backed up earlier was changed since; it was kept as it is and its copy stays next to it.";
            SplitDialogText(t, v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Close", 1, true}};
            v.focus = ID_DIALOG;
        });
        // a warning (Install anyway): no filled default, so its buttons are equals; the keyboard focus is on Cancel (RunDialog),
        // its ring shown when the keyboard is in use (dialog-warning-keys): Return cancels
        auto warning = [&](bool keys) {
            game(0);
            v.dialog.open = true; v.dialog.icon = 1;
            SplitDialogText(L"Install WrongGPU in " + v.cur.name + L" anyway?\n\n" + v.cur.name +
                                L" hasn't been tested with WrongGPU yet. It may not work, or the game may use AMD FSR instead.", v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Cancel", 0}, {L"Install anyway", 1, true}};
            v.focus = ID_DIALOG; v.focusVisible = keys;
        };
        frame(L"dialog-warning", [&] { warning(false); });
        frame(L"dialog-warning-keys", [&] { warning(true); });
        frame(L"dialog-note", [&] {  // a note (Inform) with a title alone: its one button is the default (the accent); nothing scrolls
            game(0);
            v.dialog.open = true; v.dialog.icon = 2;
            SplitDialogText(L"There's no log yet. It appears after the game runs with WrongGPU.", v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Close", 1, true}};
            v.focus = ID_DIALOG; v.focusVisible = false;
        });
        frame(L"dialog-logs-games", [&] {  // Collect logs from the launcher's uninstall question: one file per game with WrongGPU
            v.page = P_SETTINGS;
            v.dialog.open = true; v.dialog.icon = 2;
            std::vector<wstring> zips, names;
            for (size_t i = 0; i < v.games.size() && zips.size() < 3; ++i)
                if (v.games[i].hasRecord) { zips.push_back(L"C:\\Users\\Player\\Desktop\\WrongGPU-logs-" + Stem(v.games[i].exe) + L"-20261101-201455.zip"); names.push_back(v.games[i].name); }
            if (zips.size() < 2) { zips = {L"C:\\Users\\Player\\Desktop\\WrongGPU-logs-a.zip", L"C:\\Users\\Player\\Desktop\\WrongGPU-logs-b.zip"}; names = {L"Control", L"Quarry Nine"}; }
            SplitDialogText(LogsSavedText(zips, names, true), v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Done", 0}, {L"Show in File Explorer", 1, true}};
            v.focus = ID_DIALOG + 1; v.focusVisible = false;
        });
        frame(L"dialog-logs", [&] {  // Collect logs' result (LogsSaved)
            game(0);
            v.dialog.open = true; v.dialog.icon = 2;
            SplitDialogText(LogsSavedText(L"C:\\Users\\Player\\Desktop\\WrongGPU-logs-" + Stem(v.cur.exe) + L"-20261101-201455.zip", true), v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Done", 0}, {L"Show in File Explorer", 1, true}};
            v.focus = ID_DIALOG + 1; v.focusVisible = false;
        });
        // motion, halfway (live: 90 to 120 ms into an entrance): the dialog's sheet over its dimming page; a menu and a toast
        frame(L"motion-dialog", [&] {
            game(0);
            v.dialog.open = true; v.dialog.icon = 0;
            SplitDialogText(L"Uninstall WrongGPU from " + v.cur.name + L"?\n\nThe game's original files are put back exactly as they were.", v.dialog.title, v.dialog.body);
            v.dialog.buttons = {{L"Cancel", 0}, {L"Uninstall", 1, true, true}};
            v.motionAt = 0.5f;
        });
        frame(L"motion-menu", [&] { game(0); RenderFrame((float)w, (float)h, s, L""); OpenMoreMenu(); v.toast.text = L"Release address saved."; v.toast.tone = 1; v.motionAt = 0.5f; });
    }
    DestroyWindow(v.hwnd);
    if (failed) { *why = std::to_wstring(failed) + L" frame(s) not written"; return EXIT_FAILED; }
    return EXIT_OK;
}
}  // namespace wg::app
