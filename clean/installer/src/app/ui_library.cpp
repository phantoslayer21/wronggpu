// SPDX-License-Identifier: MIT
// WrongGPU launcher window: the Library page (ui_shell.h), the owner's "BE-wide" design (build/launcher/ui-options/BE-wide:
// index.html ?v=a, notes.md): a large title and its line, Search, an icon-only Rescan and Add game at the right; a featured
// banner (the game last played with WrongGPU on, else the first that needs a repair or an update: its hero art, logo, one
// line, its status, Play or its action and Game settings); then the games on shelves stacked one under the other: "Needs
// attention" (a caption, and the red capsule "Update 4 games" on its heading), "With WrongGPU", "Available", each a
// wrapping row of 16:9 tiles (the game's hero art with its logo, its name and one useful line under it, its action always
// in view as a pill beside them; a busy game shows its progress there), and at the bottom the games WrongGPU stays out of
// as one compact row each. No filter: the shelves are the grouping, the search filters. Each tile is two stops for Tab and
// the arrow keys (it opens the game's page; its action). Written for this project.
#include "app/ui_shell.h"

namespace wg::app::lui {

namespace {
TS Ts(float size, int weight, Col c, FontFace f = F_TEXT) { TS s; s.size = size; s.weight = weight; s.color = c; s.face = f; return s; }

// A game of the running update queue (Update all): it keeps its place in "Needs attention" until the queue ends, so the
// lockups do not move under the pointer as each game finishes.
bool InQueueRun(const GameEntry& e) {
    if (v.queueTotal <= 0) return false;
    for (const wstring& d : v.queueDirs) if (IEq(d, e.dir)) return true;
    return false;
}

// (an install whose game needs a tool of the player's that is gone since, REFramework: WrongGPU can't run there)
bool NeedsTool(const GameEntry& e) { return e.hasRecord && e.status == GameStatus::NeedsPrereq; }
bool Attention(const GameEntry& e) {
    return e.status == GameStatus::UpdateAvailable || e.status == GameStatus::NeedsRepair || e.status == GameStatus::Orphaned || e.adoptable ||
           e.status == GameStatus::Leftovers || NeedsTool(e) || RrConfigReadyToAdd(e);
}
// An install the launcher does not manage yet (set up by WrongGPU setup 0.5, without profile.ini, or with its own uninstall
// entry once the launcher is installed): Adopt brings it in, its settings kept. A game with an update or a repair gets
// that instead (it moves it to the launcher too).
bool ToAdopt(const GameEntry& e) {
    return e.adoptable && (e.status == GameStatus::Installed || e.status == GameStatus::TurnedOff || e.status == GameStatus::Running);
}
// A game WrongGPU can go into (a running game too: it waits on the shelf with its Install until the game closes).
bool Available(const GameEntry& e) {
    return !e.hasRecord && (e.status == GameStatus::Eligible || e.status == GameStatus::PlayersOptiScaler || e.status == GameStatus::OldInstall ||
                            e.status == GameStatus::Leftovers || e.status == GameStatus::Running || e.status == GameStatus::NeedsPrereq);
}

}  // namespace

// Borderlands 4 / Hogwarts Legacy in a protected folder, Fix it for me chosen, the game's settings folder there now: the
// Engine.ini line waits for the player's click (the window never adds it by itself there, r2c). Shown in "Needs attention"
// with Fix it now, so a player who just plays learns it (r3 #5). An update or a repair adds it too: those say so themselves.
// The line is Ray Reconstruction's, so RR's own switch decides (r5 #5): with RR turned off (rr_enabled off) nothing needs the
// player's attention there; with the upscaler turned off (Use WrongGPU off: TurnedOff) RR still runs, so it is listed. Only an
// install with WrongGPU's RR files (r6 #5: one without them, offline or not the WrongGPU edition, has no RR of WrongGPU's for the
// line to serve: no attention item, toast or permission prompt for it)
bool RrConfigReadyToAdd(const GameEntry& e) {
    return e.hasRecord && e.rrInstalled && e.rrConfigDue && e.rrConfigProtected && !e.adoptable && e.rrEnabled &&
           (e.status == GameStatus::Installed || e.status == GameStatus::TurnedOff);
}

// The plates' hues: ten well apart (36 degrees). A game's plate has its own, from a hash of its name only, so it keeps its
// colour wherever it is shown: on any shelf, in a search, after the shelves reorder, and on its own page.
constexpr int kPlateHues = 10;
static int NameHue(const wstring& name) {
    unsigned hsh = 2166136261u;
    for (wchar_t c : name) hsh = (hsh ^ (unsigned)c) * 16777619u;
    return (int)(hsh % kPlateHues);
}

// A stable accent colour from the name (generated art), and a dark partner.
void NameColours(const wstring& name, Col& a, Col& b) {
    const float hue = (NameHue(name) * (360.f / kPlateHues) + 12.f) / 360.f;
    auto hsl = [](float h, float s, float l) {
        auto f = [&](float n) {
            const float k = std::fmod(n + h * 12, 12.f), aa = s * std::min(l, 1 - l);
            return l - aa * std::max(-1.f, std::min({k - 3, 9 - k, 1.f}));
        };
        return Col{f(0), f(8), f(4), 1};
    };
    a = hsl(hue, 0.55f, 0.55f);
    b = hsl(hue, 0.35f, 0.18f);
}

// Art for a game without Steam art, in the bench's style: a graphite plate (the same in both themes) with a faint brushing
// and a glow in the name's colour; a bay's strip also gets an accent bar on its left edge. The name is drawn over it by
// the caller.
void GeneratedArt(Gfx& gx, const Rc& r, const wstring& name, bool bar) {
    Col accent, dark; NameColours(name, accent, dark);
    gx.Grad(r, 0, pal.genA, pal.genB, 0, 0.1f, 1, 0.9f);
    gx.Glow(r.x + r.w * 0.85f, r.y, r.w * 0.5f, r.h * 1.1f, Alpha(accent, 0.10f));
    gx.PushClip(r);
    for (float x = r.x; x < r.r(); x += 3) gx.Fill(Rc{x, r.y, 1, r.h}, Alpha(pal.onArt, 0.018f));
    gx.PopClip();
    if (bar) gx.Fill(Rc{r.x, r.y, 3, r.h}, Alpha(accent, 0.85f));
}

// A path that does not fit, shortened in the middle: "G:\SteamLibrary\steamapps\…\Cyberpunk 2077\bin\x64".
wstring FitMiddle(Gfx& gx, const wstring& t, float width, const TS& s) {
    if (t.size() < 6 || gx.TextW(t, s) <= width) return t;
    size_t lo = 2, hi = t.size() - 1;
    auto cut = [&](size_t keep) { const size_t head = keep * 2 / 5; return t.substr(0, head) + L"\u2026" + t.substr(t.size() - (keep - head)); };
    while (lo < hi) {  // the most characters kept that still fit
        const size_t mid = (lo + hi + 1) / 2;
        if (gx.TextW(cut(mid), s) <= width) lo = mid; else hi = mid - 1;
    }
    return cut(lo);
}

// The game a converted model file came from: the name of the game whose folder holds the file (the store libraries or
// the Library), else the Steam folder name, else the file's folder. Empty when the path says nothing useful.
wstring ModelSourceGame(const wstring& src, const std::vector<StoreGame>* libs) {
    const wstring low = Lower(src);
    const size_t slash = low.find_last_of(L"\\/");
    if (slash == wstring::npos || slash == 0) return wstring();
    const wstring folder = low.substr(0, slash + 1);
    std::vector<std::pair<wstring, wstring>> known;  // (folder, name)
    if (libs) for (const StoreGame& s : *libs) known.push_back({s.root, s.name});
    else for (const GameEntry& e : v.games) known.push_back({e.dir, e.name});
    for (const auto& k : known) {
        if (k.first.empty()) continue;
        const wstring d = Lower(k.first) + L"\\";
        if (folder.rfind(d, 0) == 0 || d.rfind(folder, 0) == 0) return k.second;
    }
    const wstring key = L"\\steamapps\\common\\";
    const size_t p = low.find(key);
    if (p != wstring::npos) {
        const size_t b = p + key.size(), end = src.find_first_of(L"\\/", b);
        if (end != wstring::npos) {
            const wstring top = low.substr(0, end + 1);
            for (const auto& k : known)
                if (!k.first.empty() && (Lower(k.first) + L"\\").rfind(top, 0) == 0) return k.second;
            return src.substr(b, end - b);
        }
    }
    // else the game's folder: above bin\x64 or Binaries\Win64 when the file sits there ("Quarry Nine", not "x64")
    const wstring root = GameRoot(src.substr(0, slash));
    const size_t prev = root.find_last_of(L"\\/");
    return prev == wstring::npos ? wstring() : root.substr(prev + 1);
}

// The record's problems in plain words: "changed: x" / "missing: x" / "wronggpu.ini is missing".
wstring ProblemSentence(const wstring& p) {
    if (p.rfind(L"changed: ", 0) == 0) return p.substr(9) + L" was changed after the install.";
    if (p.rfind(L"missing: ", 0) == 0) return p.substr(9) + L" is missing.";
    if (p == L"wronggpu.ini is missing") return L"WrongGPU's settings file is missing.";
    return p + L".";
}
// The count in the game page's form ("2 of WrongGPU's files are missing or changed", "One of WrongGPU's files is missing"):
// one way of saying it on the banner, the attention line, the lockup's accessible name and the page.
wstring ProblemShort(const GameEntry& e) {
    if (!e.realExe.empty()) return L"WrongGPU is in the wrong place";  // next to the game's launcher or starter (rc6 L2)
    if (e.problems.empty()) return L"Needs a repair";
    if (e.problems.size() > 1) {
        size_t changed = 0;
        for (const wstring& p : e.problems) changed += p.rfind(L"changed: ", 0) == 0;
        return std::to_wstring(e.problems.size()) + L" of WrongGPU's files " +
               (changed == 0 ? L"are missing" : changed == e.problems.size() ? L"were changed" : L"are missing or changed");
    }
    const wstring& p = e.problems[0];
    if (p.rfind(L"changed: ", 0) == 0) return L"One of WrongGPU's files was changed";
    if (p.rfind(L"missing: ", 0) == 0) return L"One of WrongGPU's files is missing";
    if (p == L"wronggpu.ini is missing") return L"WrongGPU's settings file is missing";
    return p.size() < 34 ? p : L"Needs a repair";
}

int NeedsAttentionCount() { int n = 0; for (auto& e : v.games) n += Attention(e); return n; }
// the games that need attention for something other than an update (a repair, files left behind, a game gone): the tab's
// count is red only while one of these is there (UI review #3). An Engine.ini line ready to add is routine too, as an update
// is (its action is Fix it now on the accent's tint, nothing is broken): never red on its own (r4 #8)
bool RedAttention(const GameEntry& e) { return Attention(e) && e.status != GameStatus::UpdateAvailable && !RrConfigReadyToAdd(e); }
int ProblemCount() { int n = 0; for (auto& e : v.games) n += RedAttention(e); return n; }
int UpdateCount() { int n = 0; for (auto& e : v.games) n += e.status == GameStatus::UpdateAvailable; return n; }

wstring ExecutorName(const wstring& e) { return e == L"dx12" ? L"D3D12" : e == L"hip" ? L"HIP" : e; }

wstring StoreOf(const GameEntry& e) {
    if (e.source == L"Steam" || e.source == L"Epic" || e.source == L"GOG") return e.source;
    if (!e.appid.empty() || Lower(e.exe).find(L"\\steamapps\\") != wstring::npos) return L"Steam";
    if (Lower(e.exe).find(L"\\epic games\\") != wstring::npos) return L"Epic";
    if (Lower(e.exe).find(L"\\gog") != wstring::npos) return L"GOG";
    return e.source == L"Added" ? L"Added by you" : L"Your folder";
}

wstring ShortStatus(const GameEntry& e) {
    switch (e.status) {
        case GameStatus::Installed: return L"Installed " + e.installedVersion;
        case GameStatus::TurnedOff: return L"Off \u00b7 AMD FSR upscales";
        case GameStatus::UpdateAvailable: return InstalledLabel(e) + L" \u2192 " + Ctx().data->version;
        case GameStatus::NeedsRepair: return ProblemShort(e);
        case GameStatus::Orphaned: return L"The game is gone \u00b7 Remove files cleans up";
        case GameStatus::Running: return L"Running";
        case GameStatus::Eligible: return StoreOf(e) + (e.statusText.find(L"experimental") != wstring::npos ? L" \u00b7 experimental" : L"");
        case GameStatus::PlayersOptiScaler: return L"Install asks to switch";
        case GameStatus::OldInstall: return L"Has an older WrongGPU";
        case GameStatus::Leftovers: return L"Leftover WrongGPU files";
        case GameStatus::NeedsPrereq: return L"Needs " + e.needs;
        case GameStatus::Refused: {
            wstring s = e.statusText.rfind(L"Refused: ", 0) == 0 ? e.statusText.substr(9) : e.statusText;
            if (Lower(s).rfind(L"anti-cheat (", 0) == 0) {  // the tile names the anti-cheat; its marker path is on the game page
                wstring m = Lower(s);
                const size_t leaf = m.find_last_of(L"\\/");
                if (leaf != wstring::npos) m = m.substr(leaf + 1);  // the marker's own name, not the game's folders
                static const std::pair<const wchar_t*, const wchar_t*> kNames[] = {
                    {L"easyanticheat", L"Easy Anti-Cheat"}, {L"start_protected_game", L"Easy Anti-Cheat"}, {L"battleye", L"BattlEye"},
                    {L"beservice", L"BattlEye"}, {L"eaanticheat", L"EA anti-cheat"}, {L"gameguard", L"GameGuard"}, {L"nprotect", L"GameGuard"},
                    {L"vgk.sys", L"Vanguard"}, {L"vgc.exe", L"Vanguard"}, {L"pnkbstr", L"PunkBuster"}, {L"xigncode", L"XIGNCODE3"},
                    {L"randgrid", L"Ricochet"}, {L"neac", L"NetEase anti-cheat"}, {L"anticheatexpert", L"Anti-Cheat Expert"},
                    {L"ace-base", L"Anti-Cheat Expert"}, {L"sguard64", L"Anti-Cheat Expert"}, {L"tenprotect", L"Anti-Cheat Expert"},
                    {L"anogs", L"Anti-Cheat Expert"}, {L"mhyprot", L"mhyprot"}};
                for (const auto& n : kNames) if (m.find(n.first) != wstring::npos) return wstring(L"Anti-cheat: ") + n.second;
                return L"Anti-cheat";
            }
            if (!s.empty()) s[0] = towupper(s[0]);
            return s;
        }
        case GameStatus::ForeignMod: return L"Other upscaler mod";
    }
    return e.statusText;
}

// The status LED of a game: green WrongGPU working, amber needs a look, grey off.
Col StatusLed(const GameEntry& e, bool& lit) {
    lit = true;
    switch (e.status) {
        case GameStatus::Installed: case GameStatus::Running: return e.adoptable || RrConfigReadyToAdd(e) ? pal.amber : pal.green;
        case GameStatus::TurnedOff: if (RrConfigReadyToAdd(e)) return pal.amber; lit = false; return pal.off;  // (r5 #5: its RR line waits)
        case GameStatus::UpdateAvailable: case GameStatus::NeedsRepair: case GameStatus::Orphaned: case GameStatus::OldInstall: case GameStatus::Leftovers:
            return pal.amber;
        default: lit = false; return pal.off;
    }
}

// How urgent a game's problem is: "Needs attention" lists a broken install first, then a game that is gone, leftovers to
// look at, then updates (a game the running queue has updated stays with them).
static int Urgency(const GameEntry& e) {
    if (InQueueRun(e)) return 3;
    switch (e.status) {
        case GameStatus::NeedsRepair: return 0;
        case GameStatus::Orphaned: return 1;
        case GameStatus::UpdateAvailable: return 3;
        default: return 2;
    }
}

// ---- the lockups (16:9 tiles) -------------------------------------------------------------------------------------------------
namespace {
constexpr float kArtR = 12;                           // a tile's art: 16:9, radius 12
constexpr float kLockGap = 24, kLockRowGap = 30;      // between tiles; between rows of tiles
constexpr float kLockPillH = 26, kLockPillMinW = 68;  // a tile's action (its hit 28 high: 1 DIP more above and below)
constexpr float kLockHitH = 28;
constexpr float kMetaGap = 10, kNameH = 19, kLineGap = 2, kLineH = 17;  // under the art: the name (14), its line (12.5)
constexpr float kBusyW = 96;                          // a busy tile's progress, in its pill's place
// This frame's grid (PaintLibrary): 4 tiles a row in the column, 3 below about 860 DIPs, 2 at the smallest window; they
// span the column, as its tools, captions and hairlines do. Under the art: the name and its line on the left, the action
// pill on the right of that same block.
struct Grid { int n = 4; float artW = 252, artH = 142, gap = kLockGap; };
Grid grid;
float MetaY() { return grid.artH + kMetaGap; }
float LineY() { return MetaY() + kNameH + kLineGap; }
float TileH() { return LineY() + kLineH; }
Grid GridFor(float cw) {
    Grid gd;
    gd.n = cw >= 860 ? 4 : cw >= 700 ? 3 : 2;
    gd.artW = std::floor((cw - (gd.n - 1) * kLockGap) / gd.n);
    gd.artH = std::round(gd.artW * 9 / 16);
    gd.gap = gd.n > 1 ? (cw - gd.n * gd.artW) / (gd.n - 1) : kLockGap;
    return gd;
}
constexpr float kShelfGap = 34, kRuleGap = 30;        // a shelf's margin above; under the hairline between shelves
constexpr float kShelfHeadH = 30, kTilesGap = 16;     // a shelf's heading row (the capsule's height); heading (or caption) to the art
constexpr float kRefRowH = 45, kRefCoverW = 30;       // "Not supported": one compact row per game
constexpr int kAttnRows = 2;                          // "Needs attention" shows two rows of tiles; more wait behind "Show 12 more"
// the featured banner: 300 high, radius 16; lower in a short window (FeatureBanner), so that at the smallest window the
// first screen shows the shelf under it too (SHIP-BAR 3)
constexpr float kBannerMaxH = 300, kBannerMinH = 240, kBannerR = 16;
constexpr float kHeroKeep = 0.94f;  // the part of a hero's height the banner may show, from its top (HeroCover: no dark band)

// Where the tiles and the compact rows were in the last frame (content DIPs, reading order), with the shelves' own buttons
// and the banner's between them ("Update 4 games", "Show 12 more" / "Show fewer", Play: id set, index -1): the arrow keys
// walk them.
struct Slot { int index; Rc r; int id = 0; int shelf = -1; };  // shelf: 0..3 for a lockup or a compact row (the anchor's)
std::vector<Slot> slots;
}  // namespace

// Where a game comes from, inside a sentence: " from Steam", " from your folder", ", added by you".
static wstring FromWhere(const GameEntry& e) {
    const wstring s = StoreOf(e);
    if (s == L"Added by you") return L", added by you";
    return L" from " + (s == L"Your folder" ? wstring(L"your folder") : s);
}

// What a lockup says to UI Automation after the game's name: its state in a plain-language sentence or two, more than the
// one line under the art (the version, when it ran and on what).
static wstring LockupStatus(const GameEntry& e) {
    const wstring ver = Ctx().data->version;
    const bool experimental = e.statusText.find(L"experimental") != wstring::npos;
    switch (e.status) {
        case GameStatus::Installed: {
            wstring t = L"WrongGPU " + e.installedVersion + L" installed.";
            if (ToAdopt(e)) return t + L" Set up outside the launcher: Adopt lets the launcher manage it and keeps its settings.";
            if (RrConfigReadyToAdd(e))  // r3 #5
                return t + L" Its Ray Reconstruction setting is ready to add to Engine.ini: Fix it now adds it. Windows asks for permission first.";
            const wstring on = e.last.executor.empty() ? L"" : L" on " + wstring(e.last.executor == L"dx12" ? L"Direct3D 12" : L"HIP");
            // the page's order (r7 #6): RR doing the upscaling (an RR-only run with frames, or a mixed run where Model M upscaled none)
            // and an RR attempt without a successful frame come before Model M's ready, as the Upscaler and Denoiser rows tell it
            const bool rrFailed = e.last.found && e.last.rrUsed && e.last.rrFrames <= 0;  // (r6 #2)
            const wstring noRrFrame = L"no successful Ray Reconstruction frame recorded.";
            if (e.last.found && (e.last.rrUpscaled || (e.last.rrOnly && e.last.upscaled)))  // r4 #1, r6 #1
                t += L" Last run " + When(e.last.when) + L" with Ray Reconstruction.";
            else if (rrFailed && !e.last.ready && e.last.upscaler != L"amd")  // RR-only, or beside a Model M the game didn't ask (r7 #1)
                t += L" Last run " + When(e.last.when) + L": " + noRrFrame;
            else if (e.last.found && e.last.ready) t += L" Last run " + When(e.last.when) + on + (rrFailed ? L"; " + noRrFrame : wstring(L"."));
            else if (e.last.found && e.last.upscaler == L"amd") t += L" Last run " + When(e.last.when) + L" with AMD FSR" + (rrFailed ? L"; " + noRrFrame : wstring(L"."));
            else if (!e.last.found) t += L" Not played yet.";  // the lockup's own words
            return t;
        }
        case GameStatus::TurnedOff:
            if (RrConfigReadyToAdd(e))  // RR runs with the upscaler off (r5 #5)
                return e.installedVersion + L" installed, turned off: AMD FSR upscales. Its Ray Reconstruction setting is ready to add to Engine.ini: Fix it now adds "
                                            L"it. Windows asks for permission first.";
            return e.installedVersion + L" installed, turned off: AMD FSR upscales.";
        case GameStatus::UpdateAvailable: return L"Update ready: " + InstalledLabel(e) + L" → " + ver + L".";
        case GameStatus::NeedsRepair: {
            if (!e.realExe.empty()) return L"Needs a repair: WrongGPU is in the wrong place. Repair moves it to the game.";  // rc6 L2
            if (e.problems.size() == 1) {  // the game page's words (its Game folder row): no file names, which its tooltip has
                const wstring& p = e.problems[0];
                if (p.rfind(L"changed: ", 0) == 0) return L"Needs a repair: one of WrongGPU's files was changed.";
                if (p.rfind(L"missing: ", 0) == 0) return L"Needs a repair: one of WrongGPU's files is missing.";
                if (p == L"wronggpu.ini is missing") return L"Needs a repair: WrongGPU's settings file is missing.";
            }
            if (e.problems.size() > 1) return L"Needs a repair: " + ProblemShort(e) + L".";  // "2 of WrongGPU's files are missing or changed"
            return ProblemShort(e) + L".";
        }
        case GameStatus::Orphaned: return L"The game is gone. Remove files on its page deletes what WrongGPU left in its folder.";
        case GameStatus::Running:
            return e.hasRecord ? L"Running now. Install, Update and settings wait until it closes." : L"Running now. Install waits until it closes.";
        case GameStatus::Eligible:
            return L"Ready to install" + FromWhere(e) + L". " + (experimental ? L"Not tested with WrongGPU yet." : L"Tested with WrongGPU.");
        case GameStatus::PlayersOptiScaler: return L"Has your own OptiScaler. Install asks to switch to WrongGPU's OptiScaler (yours is backed up).";
        case GameStatus::OldInstall: return L"An older WrongGPU (DLSS 4.5 AMD) is in this game. Install replaces it.";
        case GameStatus::Leftovers: return L"Leftover WrongGPU files are here. Clean up or adopt them on its page.";
        case GameStatus::NeedsPrereq:  // "the game": the tile's name is the Library's, never a second name for it
            return L"Needs " + e.needs + (e.hasRecord ? L": WrongGPU can't load in the game without it." : L" first: WrongGPU can't load in the game without it.") + L" Its page says more.";
        case GameStatus::Refused: return ShortStatus(e) + L". WrongGPU stays out of this game.";
        case GameStatus::ForeignMod: return L"Another upscaler mod is here. WrongGPU stays out of this game.";
    }
    return e.statusText;
}
wstring LockupSaid(const GameEntry& e) { return LockupStatus(e); }

// The words for an untested game, the same on its lockup and its page
static bool Untested(const GameEntry& e) { return e.status == GameStatus::Eligible && e.statusText.find(L"experimental") != wstring::npos; }

// "Played Oct 17" ("Oct 17 at 19:19" -> "Oct 17"; "Today at 19:19" -> "today"), "Not played yet"
static wstring PlayedOn(const GameEntry& e) {
    if (!e.last.found) return L"Not played yet";
    wstring d = When(e.last.when);
    const size_t at = d.find(L" at");
    if (at != wstring::npos) d = d.substr(0, at);
    if (d == L"Today" || d == L"Yesterday") d[0] = towlower(d[0]);
    return L"Played " + d;
}

// A lockup's one useful line (12, secondary): what is wrong (amber, semibold), the update's versions (its pill says Update),
// when it was played (the tint's words, semibold: WrongGPU is on in this game), why it is on its shelf. The words that
// matter come first; where the game comes from follows after a middle dot (`store`: on every tile of the shelf or on
// none, MeasureLockup; it is always in the tooltip and the accessible name).
static wstring LockupLine(const GameEntry& e, bool& amber, bool& on, wstring& store) {
    amber = false;
    on = e.status == GameStatus::Installed || (e.status == GameStatus::Running && e.hasRecord);
    store.clear();
    if (ToAdopt(e) && e.status != GameStatus::Running) { amber = true; on = false; return L"Not managed yet"; }
    if (RrConfigReadyToAdd(e)) { amber = true; on = false; return L"Engine.ini: ready to add"; }  // the card's words (r3 #5)
    switch (e.status) {
        case GameStatus::NeedsRepair: amber = true; return L"Needs a repair";
        case GameStatus::Orphaned: amber = true; return L"The game is gone";
        case GameStatus::Leftovers: amber = true; return L"Leftover files";
        case GameStatus::NeedsPrereq: amber = true; return L"Needs " + e.needs;  // the page's words (a warning: amber, never red)
        case GameStatus::UpdateAvailable: return InstalledLabel(e) + L" → " + wstring(Ctx().data->version);
        case GameStatus::Running: return L"Running now";
        case GameStatus::TurnedOff: return L"Turned off";
        case GameStatus::Installed: {
            if (InQueueRun(e)) return L"Updated to " + e.installedVersion;  // done in the running Update all: it keeps its place
            return PlayedOn(e);
        }
        // what the game has, as a sentence without its subject (the accessible name and ShortStatus say it the same way)
        case GameStatus::PlayersOptiScaler: store = StoreOf(e); return L"Has your own OptiScaler";  // the game page's tag
        case GameStatus::OldInstall: store = StoreOf(e); return L"Has an older WrongGPU";
        default:
            if (Untested(e)) { store = StoreOf(e); return L"Not tested yet"; }
            return StoreOf(e);
    }
}

// Whether an installed game's lockup (and the banner) offers Play, else Details: Settings > Play button on and its .exe found
// by the scan (or the saved list). The pill's label and what pressing it does (launcher_ui.cpp's Activate) both read this,
// so a pill never does something other than what it says; an .exe gone since is Play's own note ("The game's program is
// missing...").
bool LockupPlays(const GameEntry& e) { return gl.global.playButton && (v.fixture || e.exeFound); }

// A repair of an older WrongGPU is an update too (launcher_ui.cpp's question "Repair and update", its busy words "Repairing
// and updating...", the result "is repaired and updated"): the lockup's pill and the banner's button name it so from the
// start, as the game page's button does (ui_game.cpp RepairUpdatesHere: the same rule).
static bool RepairUpdatesToo(const GameEntry& e) {
    return UpdateDue(e, Ctx());  // an older version, or another build of this one (rc5)
}

// A lockup's action, always in view: Repair, Adopt and Review on the red tint (the only red on a lockup: something is broken
// or unclaimed and needs you), Play and Update on the accent's tint (the player's pick: an update is routine, UI review #3),
// Install on the grey fill. A game with WrongGPU but no
// Play (Settings > Play button off, or its .exe not found) gets Details, which opens its page; Review opens the page of a
// game that is gone or has leftovers (Uninstall, Clean up, Adopt are there). Id: ID_TILEQA + index (launcher_ui.cpp's Activate).
static wstring LockupAction(const GameEntry& e, bool& tint) {
    tint = false;
    if (ToAdopt(e) && e.status != GameStatus::Running) { tint = true; return L"Adopt"; }
    if (RrConfigReadyToAdd(e)) return L"Fix it now";  // the accent's tint, as Update: the player's pick, nothing broken (r3 #5)
    switch (e.status) {
        case GameStatus::Installed: case GameStatus::TurnedOff: return LockupPlays(e) ? L"Play" : L"Details";
        case GameStatus::Running: return e.hasRecord ? L"Play" : L"Install";  // disabled until the game closes
        case GameStatus::UpdateAvailable: return L"Update";
        case GameStatus::NeedsRepair: tint = true; return RepairUpdatesToo(e) ? L"Repair and update" : L"Repair";
        case GameStatus::Orphaned: case GameStatus::Leftovers: tint = true; return L"Review";
        case GameStatus::NeedsPrereq: return L"Details";  // Install is held back: its page says what the game needs and where to get it
        case GameStatus::Eligible: case GameStatus::PlayersOptiScaler: case GameStatus::OldInstall: return L"Install";
        default: return L"";
    }
}

// A game of the running Update all that has not started yet.
static bool Waiting(const GameEntry& e) {
    return InQueueRun(e) && e.status == GameStatus::UpdateAvailable && !(v.busy && IEq(v.busyDir, e.dir));
}
// What a busy game's lockup says in its action's place: "Updating... 62%", "Repairing...", "Installing...".
static wstring BusyLabel(const GameEntry& e) {
    wstring t = v.busyVerb;
    if (t.empty()) {  // the operation's verb is set when it starts; a fixture frame has only the game's state
        switch (e.status) {
            case GameStatus::UpdateAvailable: t = L"Updating…"; break;
            case GameStatus::NeedsRepair: t = RepairUpdatesToo(e) ? L"Repairing and updating…" : L"Repairing…"; break;
            case GameStatus::Eligible: case GameStatus::PlayersOptiScaler: case GameStatus::OldInstall: t = L"Installing…"; break;
            case GameStatus::Orphaned: t = L"Removing files…"; break;  // its button's verb (Remove files)
            case GameStatus::Leftovers: t = L"Cleaning up…"; break;
            default: t = L"Working…"; break;
        }
    }
    if (v.progress > 0) t += L" " + std::to_wstring((int)std::lround(std::min(1.f, v.progress) * 100)) + L"%";
    return t;
}

// A lockup's accessible name (the stop that opens the game's page): the game and what it is doing or needs.
static wstring LockupName(const GameEntry& e, bool dupe) {
    if (v.busy && IEq(v.busyDir, e.dir)) {
        const wstring b = BusyLabel(e);  // "Repairing…" (no period after its ellipsis), "Repairing… 62%"
        return e.name + L" - " + b + (v.step.empty() || b.rfind(v.step, 0) == 0 ? L"" : (!b.empty() && b.back() == L'…' ? L" " : L". ") + v.step);
    }
    if (Waiting(e)) return e.name + L" - Waiting to update.";  // its turn in the running "Update N games"
    // a game whose name another game of the Library has too (a folder's copy of a Steam game): where it comes from, as
    // its tile's line says it ("Cyberpunk 2077 from your folder - ..."), so Narrator tells the two apart as well
    const wstring status = LockupStatus(e);
    if (dupe && status.find(L" from ") == wstring::npos) return e.name + FromWhere(e) + L" - " + status;
    return e.name + L" - " + status;
}

// The short name a plate shows: "Tom Clancy's Splinter Cell: Chaos Theory - Remastered Collector's Edition" without its
// edition after " - " (the whole name is under the art).
static wstring PlateName(const wstring& n) {
    const size_t d = n.find(L" - ");
    return d != wstring::npos && d >= 4 ? n.substr(0, d) : n;
}

// A plate's short name drops a publisher's name in front too: "Tom Clancy's Splinter Cell: Chaos Theory" -> "Splinter
// Cell: Chaos Theory" (two capitalised words, the second ending in 's; "Marvel's" or "Assassin's Creed" stay).
static wstring PlateTitle(const wstring& n) {
    wstring s = PlateName(n);
    const size_t sp = s.find(L' ');
    if (sp == wstring::npos || sp == 0 || !iswupper(s[0]) || sp + 1 >= s.size() || !iswupper(s[sp + 1])) return s;
    const size_t sp2 = s.find(L' ', sp + 1);
    if (sp2 == wstring::npos || sp2 < sp + 4 || s.size() - sp2 - 1 < 8) return s;
    const wchar_t q = s[sp2 - 2];
    return (q == L'\'' || q == L'’') && s[sp2 - 1] == L's' ? s.substr(sp2 + 1) : s;
}

// A plate's title on its lines: a number or a "(2)" stays with the word before it ("Alan Wake 2 (6)"), and a title that
// takes two lines is broken where they come out most even ("Alan / Wake 2 (6)", not "Alan Wake / 2 (6)"). When the
// number cannot stay ("Cyberpunk 2077" is wider than the plate on one line) it goes to the next line rather than the
// title being cut to "Cyberpunk...".
static wstring PlateLines(Gfx& gx, const wstring& title, float w, const TS& s) {
    TS one = s; one.wrap = false;
    auto split = [&](const wstring& t) {
        wstring best;
        float bestW = 1e9f;
        for (size_t i = 0; i < t.size(); ++i) {
            if (t[i] != L' ') continue;
            const float a = gx.TextW(t.substr(0, i), one), b = gx.TextW(t.substr(i + 1), one);
            if (a <= w && b <= w && std::max(a, b) < bestW) { bestW = std::max(a, b); best = t.substr(0, i) + L"\n" + t.substr(i + 1); }
        }
        return best;
    };
    wstring t = title;
    for (size_t i = 1; i + 1 < t.size(); ++i)
        if (t[i] == L' ' && (iswdigit(t[i + 1]) || t[i + 1] == L'(')) t[i] = L'\u00a0';
    if (gx.TextW(t, one) <= w) return t;
    if (wstring best = split(t); !best.empty()) return best;
    if (wstring best = split(title); !best.empty()) return best;
    return t;  // three lines: left to DirectWrite (and its ellipsis)
}

// A tile's art: the game's hero (library_hero.jpg, cover-fit, its focus 30 % from the top) with its logo bottom left over
// a soft bottom scrim, else header.jpg (it has the title on it: no logo), else a quiet plate in the game's own hue (from its
// name) with its short name at the bottom left (17 Display semibold). A soft shadow and a hairline edge (light: 0.5 px black
// at 10 %; dark: 1 px white at 7 %); under the pointer or the keyboard's focus the upscale wipe (UpscaleWipe; the logo stays
// crisp above it), else 7 % lighter while the tile is under the pointer, a little darker pressed. No lift.
// The Library's own motion in the last frame it drew (a cover's fade, a progress bar's travelling segment, the scanning
// skeleton's step): Animating() keeps the window's frames coming while it moves, so it runs on the display's clock (NowMs:
// QueryPerformanceCounter) as scrolling and hover do, not on a timer. Set by a drawn frame only (a measuring pass changes
// nothing), and only while the Library is the page.
static bool libMoving = false;
bool LibraryMoving() { return libMoving && v.page == P_LIBRARY && v.live; }

// A cover that arrives after its lockup was drawn as a plate (decoding at start-up) fades in over the plate (kCoverFadeMs)
// instead of popping in; at once with Animation effects off and in still frames. The art waited for (by app id, UI thread)
// and when each began to fade in.
constexpr float kCoverFadeMs = 200;
static std::set<wstring> coverAwaited;
static std::map<wstring, ULONGLONG> coverFadeAt;
static float CoverFade(Gfx& gx, const wstring& appid, bool haveCover) {
    if (!v.live || gx.dry || appid.empty()) return 1;
    if (!haveCover) { coverAwaited.insert(appid); return 1; }
    if (!coverAwaited.count(appid)) return 1;
    if (!MotionOn()) { coverAwaited.erase(appid); return 1; }
    const ULONGLONG now = NowMs();
    ULONGLONG& at = coverFadeAt[appid];
    if (!at) at = now;
    const float t = (float)(now - at) / kCoverFadeMs;
    if (t >= 1) { coverAwaited.erase(appid); coverFadeAt.erase(appid); return 1; }
    libMoving = true;  // the next frame follows at the display's rate until the fade ends
    return std::max(0.f, t);
}

static void Plate(Gfx& gx, const Rc& r, const GameEntry& e, float rad, bool name) {
    Col accent, dark; NameColours(e.name, accent, dark);  // dark: the hue at 35 % saturation, 18 % lightness (#20343a-like)
    gx.Grad(r, rad, Mix(dark, accent, 0.10f), Mix(dark, pal.black, 0.66f), 0.33f, 0, 0.67f, 1);
    if (!name) return;
    TS ps = Ts(17, 600, pal.white, F_DISPLAY); ps.wrap = true; ps.lineH = 20; ps.trim = true;
    const float w = r.w - 28;
    const wstring n = PlateLines(gx, PlateTitle(e.name), w, ps);
    const float h = std::min(std::min(60.f, r.h - 28), gx.TextH(n, w, ps));
    gx.Text(n, Rc{r.x + 14, r.b() - 14 - h, w, h + 2}, ps);
}

static void LockupArt(Gfx& gx, const Rc& r, const GameEntry& e, int id) {
    const bool hot = Hot(id), down = Down(id);
    if (pal.dark) gx.Shadow(r, kArtR, 10, 0.48f, 2, true);  // 0 2px 10px rgba(0,0,0,.35) (the art or plate covers its inner half)
    else gx.Shadow(r, kArtR, 8, 0.10f, 2, true);             // 0 2px 8px rgba(0,0,0,.08)
    const Art a = e.appid.empty() ? Art{} : ArtFor(e.appid);
    const ImageP img = a.hero ? a.hero : a.header;
    const float focus = a.hero ? 0.3f : 0.5f;
    const float fade = CoverFade(gx, e.appid, img != nullptr);
    if (!img || fade < 1) Plate(gx, r, e, kArtR, !img && (a.done || e.appid.empty()));  // still decoding: the plate alone until the art arrives
    if (img) gx.ImageCover(img, r, kArtR, focus, fade);
    const float wipe = UpscaleWipe(gx, id, id, r, kArtR, img, focus, e.last, Focused(id), 1, false);
    // the logo over a bottom scrim (0.42 -> 0 at 55 %), at most 58 % wide and 46 % high, a small drop shadow
    // (fixed stops, the fade on the brush's opacity: one brush for every tile and frame, SHIP-BAR 1)
    if (a.hero && a.logo) gx.GradStops(r, kArtR, {{0.f, Alpha(pal.black, 0.42f)}, {0.55f, Alpha(pal.black, 0)}, {1.f, Alpha(pal.black, 0)}}, 0.5f, 1, 0.5f, 0, fade);
    WipeMarks(gx, r, wipe, e.last);  // over the scrim: a white line and light capsules (IDENTITY.md), under the logo
    if (a.hero && a.logo) {
        // lifted clear of the capsules while they show (the last run's sizes), with the wipe's fade
        const float lift = WipeSizes(e.last) ? std::round(20 * wipe * gx.scale) / gx.scale : 0.f;
        gx.ImageContain(a.logo, Rc{r.x + 12, r.b() - 10 - lift - r.h * 0.46f, r.w * 0.58f, r.h * 0.46f}, 0, 2, fade, 6);
    }
    if (down) gx.FillRound(r, kArtR, Alpha(pal.black, 0.08f));
    else if (hot && wipe <= 0) gx.FillRound(r, kArtR, Alpha(pal.white, 0.07f));
    if (pal.dark) gx.StrokeRound(r.in(-0.5f), kArtR + 0.5f, Alpha(pal.white, 0.07f), 1);
    else gx.StrokeRound(r.in(-0.25f), kArtR + 0.25f, Alpha(pal.black, 0.10f), 0.5f);
}

// The keyboard's ring round a lockup's art (Signal's third registration place): 3 DIPs of page, then the 2 px ring drawn
// twice, 2 DIPs to the left in registration red and 2 DIPs to the right in the working green, snapped to device pixels so
// the gap stays whole at every scale. The gap is drawn last, in the page's colour: the art's soft shadow does not fill it
// and the ring's halves that fall inside it are covered (red shows on the left, green on the right, both above and below).
// Both opaque (the titles' translucent registration pair is under 3:1 on the light page): the registration red without
// its alpha (4.0:1 on the page light, 5.7:1 dark) and the working green (3.7:1 light, 11:1 dark), so a focus ring is
// 3:1 or better in both themes (WCAG 1.4.11).
static void LockupFocus(Gfx& gx, const Rc& art, int id) {
    if (!Focused(id)) return;
    Col red = pal.regR; red.a = 1;
    const Col green = pal.green;
    const float s = gx.scale, px = 1 / s;
    const float w = std::max(2.f, std::round(2 * s)) * px, gap = std::max(3.f, std::round(3 * s)) * px, d = std::max(2.f, std::round(2 * s)) * px;
    const float x0 = std::floor(art.x * s) / s, y0 = std::floor(art.y * s) / s, x1 = std::ceil(art.r() * s) / s, y1 = std::ceil(art.b() * s) / s;
    // StrokeRound strokes inside its rectangle: the ring is gap..gap + w outside the art, the gap 0..gap
    const Rc ring{x0 - gap - w, y0 - gap - w, x1 - x0 + 2 * (gap + w), y1 - y0 + 2 * (gap + w)};
    gx.StrokeRound(ring.off(d, 0), kArtR + gap + w, green, w);
    gx.StrokeRound(ring.off(-d, 0), kArtR + gap + w, red, w);
    gx.StrokeRound(Rc{x0 - gap, y0 - gap, x1 - x0 + 2 * gap, y1 - y0 + 2 * gap}, kArtR + gap, pal.bg, gap);
}

// A lockup's action pill: 26 high, at least 68 wide, 12.5 semibold; Play carries the play glyph. Its hit (the pointer,
// UI Automation's rectangle) is 28 high, 1 DIP more above and below: a control is never under 28 DIPs (the line over it
// ends 8 DIPs higher).
static float LockupPillW(Gfx& gx, const wstring& text) {
    return std::max(kLockPillMinW, std::ceil(26 + gx.TextW(text, Ts(12.5f, 600, pal.text)) + (text == L"Play" ? 16.f : 0.f)));
}
static Rc LockupPillHit(const Rc& pill) { return Rc{pill.x, pill.y - (kLockHitH - kLockPillH) / 2, pill.w, kLockHitH}; }
static void LockupPill(Gfx& gx, int id, float x, float y, const wstring& text, bool tint, bool enabled, const wstring& name) {
    // the accent's tint: Play, an update (routine), and Fix it now (an Engine.ini line ready to add: the player's pick, r3 #5)
    const bool icon = text == L"Play", pick = icon || text == L"Update" || text == L"Fix it now";
    const Rc r{x, y, LockupPillW(gx, text), kLockPillH};
    const bool hot = enabled && Hot(id), down = enabled && Down(id);
    Col bg, fg;
    // accentText on the tint 5.4:1 light, 6.5:1 dark; the accent's words on its tint 4.5:1 or better (ui_palette.h); text on
    // the fill; disabled labels text3 (3:1 or better: on the accent's tint)
    if (tint) { bg = down ? Mix(pal.accentTintHover, pal.accent, 0.08f) : hot ? pal.accentTintHover : pal.accentTint; fg = pal.accentText; }
    else if (pick) { bg = down ? Mix(pal.tintBgHover, pal.tint, 0.08f) : hot ? pal.tintBgHover : pal.tintBg; fg = pal.tintText; }
    else { bg = down ? Mix(pal.fillHover, pal.text, 0.06f) : hot ? pal.fillHover : pal.fill; fg = pal.text; }
    if (!enabled) fg = pal.text3;
    gx.FillRound(r, kLockPillH / 2, bg);
    HcEdge(gx, r, kLockPillH / 2, hot, enabled);  // a Contrast theme: an edge, as every control (its fill is the window's)
    TS ts = Ts(12.5f, 600, fg); ts.valign = 1;
    const float tw = gx.TextW(text, ts);
    float tx = std::round(r.x + (r.w - tw - (icon ? 16.f : 0.f)) / 2);
    if (icon) { gx.Icon(ic::playLine, tx + 5, r.cy(), 10, fg); tx += 16; }
    gx.Text(text, Rc{tx, r.y, tw + 2, r.h}, ts);
    FocusRing(gx, r, kLockPillH / 2, id);
    AddHit(id, LockupPillHit(r), R_BUTTON, name, enabled);
}

// The busy words as the lockup's place has room for them (`room`: 120 DIPs, one line): a verb of two ("Repairing and
// updating... 62%") by its first ("Repairing... 62%"), a verb that still does not fit by its first word ("Applying
// settings... 62%" -> "Applying... 62%"), so the percentage is never cut. The whole words stay in the progress's and the
// lockup's accessible names and in the attention line ("Repairing and updating Bodycam.").
static wstring LockupBusyShown(Gfx& gx, const wstring& label, float room, const TS& s) {
    const size_t dots = label.find(L'…');
    if (dots == wstring::npos || gx.TextW(label, s) <= room) return label;
    wstring verb = label.substr(0, dots);
    const wstring rest = label.substr(dots);  // "... 62%"
    if (const size_t two = verb.find(L" and "); two != wstring::npos) verb.resize(two);
    if (gx.TextW(verb + rest, s) > room) if (const size_t sp = verb.find(L' '); sp != wstring::npos) verb.resize(sp);
    return verb + rest;
}

// A busy game's action place: a 4 DIP bar (the progress run, red to green, on the fill) over "Updating... 62%" (11.5
// secondary). Without a percentage a segment travels along the bar (Windows' Animation effects off: the whole run dimmed):
// the shared ProgressBar, as the game page's header has it. A game waiting in the running Update all: the bar's track and
// "Waiting...".
static void LockupProgress(Gfx& gx, const Rc& r, const wstring& label, bool busy) {
    ProgressBar(gx, Rc{r.x, r.y + 3, r.w, 4}, v.progress, busy);
    TS ls = Ts(11.5f, 400, pal.text2);
    const Rc lr{r.x, r.y + 11, r.w + 24, 16};
    gx.Text(LockupBusyShown(gx, label, lr.w, ls), lr, ls);
    AddProgress(0, Rc{r.x, r.y + 6, r.w, r.h - 6}, label, busy && v.progress > 0 ? v.progress : -1.f);  // for UI Automation: where the pill was
}

// A tile's words under its art, measured once a frame: the name (14 semibold, one line, an ellipsis), its one line (12.5)
// under it, both beside the action (or the progress). The line is always "what · where" in that order ("Not tested yet ·
// Steam", "Not played yet · Your folder"). Where the game comes from is on every tile of a shelf or on none
// (ShelfStores: `stores`), so the tiles side by side read alike. `dupe`: another game of the Library has the same name (a
// folder's copy of a Steam game): its line says where it comes from when that fits; when it does not, the words that
// matter come first: in full and without the store when they already tell the two apart ("Has your own OptiScaler"
// beside "Played Oct 17"), else the store in full and the words before it shortened ("Not pla... · Your folder").
struct LockupText {
    TS ns, ls;
    float textW = 0;
    bool nameCut = false, cutLine = false, dupe = false;
    wstring line, whole;
    size_t storeAt = wstring::npos;  // where " · Steam" starts in a line in amber or the tint's words: the store stays grey
};
static float LockupTextW(Gfx& gx, const GameEntry& e) {
    float side = 0;
    if ((v.busy && IEq(v.busyDir, e.dir)) || Waiting(e)) side = kBusyW;
    else { bool tint = false; const wstring a = LockupAction(e, tint); if (!a.empty()) side = LockupPillW(gx, a); }
    return std::max(40.f, grid.artW - 2 - (side > 0 ? side + 12 : 0.f));
}
static TS LockupLineStyle(bool amber, bool on) { return Ts(12.5f, amber || on ? 600 : 400, amber ? pal.amberText : on ? pal.greenText : pal.text2); }
// Whether a shelf's tiles say where each game comes from: the shelf whose tiles name it (Available: "Not tested yet ·
// Steam"), or one with a game whose name another game has too (its store tells the two apart: the other tiles say theirs
// as well); and only when every tile of the shelf has room for it.
static bool ShelfStores(Gfx& gx, const std::vector<int>& tiles, const std::map<wstring, int>& named) {
    bool asks = false;
    for (int i : tiles) {
        const GameEntry& e = v.games[i];
        bool amber = false, on = false;
        wstring store;
        LockupLine(e, amber, on, store);
        const auto it = named.find(Lower(e.name));
        asks = asks || !store.empty() || (it != named.end() && it->second > 1);
    }
    if (!asks) return false;
    for (int i : tiles) {
        const GameEntry& e = v.games[i];
        bool amber = false, on = false;
        wstring store;
        const wstring what = LockupLine(e, amber, on, store);
        if (what != StoreOf(e) && gx.TextW(what + L" · " + StoreOf(e), LockupLineStyle(amber, on)) > LockupTextW(gx, e)) return false;
    }
    return true;
}
// Whether another game of the same name says the same words under its art (two copies both "Not tested yet"): then only
// where each comes from tells them apart.
static bool NamesakeSays(const GameEntry& e, const wstring& what) {
    for (int i = 0; i < (int)v.games.size() && i < kTileSpan; ++i) {
        const GameEntry& o = v.games[(size_t)i];
        if (&o == &e || !IEq(o.name, e.name)) continue;
        bool amber = false, on = false;
        wstring store;
        if (LockupLine(o, amber, on, store) == what) return true;
    }
    return false;
}
static LockupText MeasureLockup(Gfx& gx, const GameEntry& e, bool dupe, bool stores) {
    LockupText t;
    t.dupe = dupe;
    const float tw = LockupTextW(gx, e);
    t.textW = tw;
    t.ns = Ts(14, 600, pal.text);
    t.nameCut = gx.TextW(e.name, t.ns) > tw;
    bool amber = false, on = false;
    wstring store;
    const wstring what = LockupLine(e, amber, on, store);
    // where it comes from, when the tile names it, the shelf shows it or a namesake needs it; once ("Steam" is the store)
    store = (!store.empty() || stores || dupe) && what != StoreOf(e) ? StoreOf(e) : wstring();
    t.ls = LockupLineStyle(amber, on);
    t.whole = store.empty() ? what : what + L" · " + store;
    const bool fits = gx.TextW(t.whole, t.ls) <= tw;
    if (store.empty() || (!stores && !dupe)) t.line = what;
    else if (fits) t.line = t.whole;
    else if (dupe && !NamesakeSays(e, what)) t.line = what;  // its words tell it from its namesake: they come first, whole
    else if (dupe) {  // the store in full, the words before it shortened (at least 3 letters and the ellipsis)
        const wstring tail = L"… · " + store;
        size_t keep = what.size();
        while (keep > 3 && gx.TextW(what.substr(0, keep) + tail, t.ls) > tw) --keep;
        while (keep > 0 && what[keep - 1] == L' ') --keep;
        t.line = what.substr(0, keep) + tail;
    } else t.line = what;
    // cut: the whole name, and the whole line when part of it is left out, in the tooltip (the UIA name has the state in full)
    t.cutLine = t.line != t.whole || gx.TextW(t.line, t.ls) > tw;
    // where it comes from is neutral (IDENTITY.md: store names grey): "Needs REFramework" amber, " · Steam" in text2 as
    // on every other tile
    if (!store.empty() && (amber || on)) {
        const wstring tail = L" · " + store;
        if (t.line.size() > tail.size() && t.line.compare(t.line.size() - tail.size(), tail.size(), tail) == 0) t.storeAt = t.line.size() - tail.size();
    }
    return t;
}

// A tile at (x, y): its art, name and line (one stop: the game's page) and its action beside them (the second stop), or
// its progress while busy. draw false: scrolled out of view, only the hits (Tab and UI Automation reach every tile).
static void Lockup(Gfx& gx, int index, float x, float y, const LockupText& t, bool draw) {
    const GameEntry& e = v.games[index];
    const int id = ID_TILE + index, qa = ID_TILEQA + index;
    const float aw = grid.artW;
    const Rc art{x, y, aw, grid.artH};
    const bool busy = v.busy && IEq(v.busyDir, e.dir), waiting = !busy && Waiting(e);
    if (t.nameCut || t.cutLine) v.tips[id] = t.cutLine ? e.name + L"\n" + t.whole : e.name;
    if (draw) {
        LockupArt(gx, art, e, id);
        gx.Text(e.name, Rc{x + 1, y + MetaY(), t.textW, kNameH + 2}, t.ns);
        if (t.storeAt != wstring::npos) gx.TextLead(t.line, t.storeAt, t.ls.color, t.ls.weight, Rc{x + 1, y + LineY(), t.textW, kLineH + 1}, Ts(12.5f, 400, pal.text2));
        else gx.Text(t.line, Rc{x + 1, y + LineY(), t.textW, kLineH + 1}, t.ls);
        LockupFocus(gx, art, id);
    }
    AddHit(id, Rc{x, y, aw, TileH()}, R_ITEM, LockupName(e, t.dupe), true);
    slots.push_back({index, art});
    const float ay = y + MetaY() + 1;  // the action's top: level with the name
    if (busy || waiting) {
        // the action's place keeps its id while it works, off ("Updating Lantern Row"): the keyboard focus that pressed it
        // stays there, shown off (a second Return does nothing, never opens the page), as a busy button keeps it elsewhere
        wstring verb = busy ? BusyLabel(e) : wstring(L"Waiting to update");
        if (const size_t cut = verb.find(L'…'); cut != wstring::npos) verb.resize(cut);
        AddHit(qa, LockupPillHit(Rc{x + aw - kBusyW, ay, kBusyW, kLockPillH}), R_BUTTON, verb + L" " + e.name, false);
        if (draw) LockupProgress(gx, Rc{x + aw - kBusyW, ay, kBusyW, kLockPillH}, busy ? BusyLabel(e) : wstring(L"Waiting…"), busy);
        return;
    }
    bool tint = false;
    const wstring a = LockupAction(e, tint);
    if (a.empty()) return;
    // only what conflicts waits (SHIP-BAR 1): Play waits for its own game (PlayBlocked), not for another game's operation;
    // Details and Review open the page (always); Install, Update, Repair and Adopt are operations: one at a time. A pill that
    // waits says why in its tooltip and to UI Automation ("Waits until WrongGPU finishes updating Quarry Nine.").
    const bool opens = a == L"Details" || a == L"Review";
    // (the last session's list, until the start-up scan lands: the operations wait for it, BusyWhy says so)
    const bool enabled = a == L"Play" ? !PlayBlocked(e) : opens || (!v.busy && !v.listCached && e.status != GameStatus::Running);
    const wstring name = a == L"Details" ? L"Details of " + e.name : a == L"Fix it now" ? L"Fix Engine.ini now in " + e.name : a + L" " + e.name;  // "Install Alan Wake 2"
    const float pw = LockupPillW(gx, a);
    if (draw) LockupPill(gx, qa, x + aw - pw, ay, a, tint, enabled, name);
    else AddHit(qa, LockupPillHit(Rc{x + aw - pw, ay, pw, kLockPillH}), R_BUTTON, name, enabled);
    if (!enabled) { const wstring why = BusyWhy(e); if (!why.empty()) { v.tips[qa] = why; Describe(qa, why); } }
}

// Why WrongGPU stays out of a game, as a sentence: "It draws with Vulkan, and WrongGPU needs Direct3D 12."
static wstring RefusedReason(const GameEntry& e) {
    if (e.status == GameStatus::ForeignMod) return L"Another upscaler mod is here, and WrongGPU stays out of its way.";
    wstring s = ShortStatus(e);  // "The game draws with Vulkan; WrongGPU needs Direct3D 12", "Anti-cheat: Easy Anti-Cheat"
    if (s.rfind(L"Anti-cheat: ", 0) == 0) return L"It uses " + s.substr(12) + L". WrongGPU stays out of games with anti-cheat.";
    if (s == L"Anti-cheat") return L"It uses anti-cheat. WrongGPU stays out of games with anti-cheat.";
    if (Lower(s).rfind(L"the game ", 0) == 0) s = L"It " + s.substr(9);
    for (size_t p; (p = s.find(L"; ")) != wstring::npos;) s.replace(p, 2, L", and ");
    while (!s.empty() && (s.back() == L'.' || s.back() == L' ')) s.pop_back();
    return s.empty() ? wstring(L"WrongGPU stays out of this game.") : s + L".";
}

// A game WrongGPU stays out of: one compact row (a 30 x 45 cover, the name semibold, the store and the plain reason in
// secondary text), not a poster: there is nothing to do with it here. The row opens the game's page.
static void RefusedRow(Gfx& gx, int index, float x, float y, float w, bool draw) {
    const GameEntry& e = v.games[index];
    const int id = ID_TILE + index;
    TS ns = Ts(13, 600, pal.text); ns.valign = 1;
    TS rs = Ts(13, 400, pal.text2); rs.valign = 1;
    const wstring rest = StoreOf(e) + L". " + RefusedReason(e);
    const float tx = x + kRefCoverW + 14, nw = std::min(w - (tx - x), gx.TextW(e.name, ns));
    const float rx = tx + nw + 7, rw = std::max(0.f, std::min(x + w - rx, gx.TextW(rest, rs) + 1));
    const Rc row{x, y, std::max(rx + rw, tx + nw) - x, kRefRowH};
    if (gx.TextW(e.name, ns) > nw + 0.5f || gx.TextW(rest, rs) > rw + 0.5f) v.tips[id] = e.name + L": " + rest;
    if (draw) {
        const Rc c{x, y, kRefCoverW, kRefRowH};
        Cover(gx, c, e.appid, e.name, 5, true);
        if (Down(id)) gx.FillRound(c, 5, Alpha(pal.black, 0.08f));
        else if (Hot(id)) gx.FillRound(c, 5, Alpha(pal.white, 0.07f));
        gx.Text(e.name, Rc{tx, y, nw + 1, kRefRowH}, ns);
        if (rw > 20) gx.Text(rest, Rc{rx, y, rw, kRefRowH}, rs);
        FocusRing(gx, row, 6, id);
    }
    // "Vulkan Racer from GOG. Not supported. It draws with Vulkan, and WrongGPU needs Direct3D 12." (sentences, as the
    // banner's name has them; the reason keeps its own capital, which may be a name: "WrongGPU stays out...")
    AddHit(id, row, R_ITEM, e.name + FromWhere(e) + L". Not supported. " + RefusedReason(e), true);
    slots.push_back({index, row});
}

// A shelf's heading: 20 Display semibold and its count in secondary text ("Needs attention 7"); "Not supported" a size
// smaller. A text item for UI Automation ("Needs attention, 7 games"), not a stop. Returns the title's right end.
static float ShelfHeading(Gfx& gx, int s, float x, float y, float h, float room, const wstring& title, int n, float size) {
    TS ts = Ts(size, 600, pal.text, F_DISPLAY); ts.spacing = -0.15f; ts.valign = 1;
    TS cs = Ts(size, 400, pal.text2, F_DISPLAY); cs.valign = 1;
    const wstring count = std::to_wstring(n);
    const float tw = std::min(room, gx.TextW(title, ts) + 1), cw = gx.TextW(count, cs) + 1;
    gx.Text(title, Rc{x, y, tw + 1, h}, ts);
    if (tw + 7 + cw <= room) gx.Text(count, Rc{x + tw + 7, y, cw + 1, h}, cs);
    AddHit(ID_SHELF0 + s, Rc{x, y, std::min(room, tw + 7 + cw), h}, R_TEXT, title + L", " + count + (n == 1 ? L" game" : L" games"), true, 2, L"", false);  // a heading (level 2)
    return x + tw + 7 + cw;
}

// "Needs attention": its caption (what the shelf holds, counted with the search applied) and on the heading's right the
// filled accent capsule that names what it does with a live count: "Update 4 games" (the games with an update that are not
// busy already). It is offered only while every game it would update is on the shelf, so it never acts on games the
// search hides; while the queue runs it says "Updating..." (disabled) and the caption counts the queue. Returns the height
// from the heading's top to the caption's end.
// `listed`: the group's lockups on the shelf (the featured game is on the banner, not here). The heading counts the whole
// group as the line, the button and the toolbar's Library badge do (one number for "needs attention" on the page: a
// repair on the banner is one of them); the caption says where the one that is not on the shelf is, for the eye and for
// UI Automation alike ("... Lantern Row is on the banner above.").
static float AttentionHead(Gfx& gx, const std::vector<int>& group, const std::vector<int>& listed, float x0, float y, float cw) {
    const bool running = v.queueTotal > 0;
    int updates = 0, repairs = 0, gone = 0, left = 0, adopt = 0, tool = 0, ready = 0, readyAll = 0, rrcfg = 0;
    wstring toolName;  // the tool those games need (REFramework)
    wstring only;  // the one update's game
    // a game an operation works in right now (not the queue: it has its own sentence) is named first and left out of the
    // counts, so they are the button's ("Updating Hogwarts Legacy. WrongGPU 0.5.5 is ready for 4 more games")
    const GameEntry* doing = nullptr;
    if (!running && v.busy) for (int i : group) if (IEq(v.busyDir, v.games[i].dir)) doing = &v.games[i];
    for (int i : group) {
        const GameEntry& e = v.games[i];
        if (&e == doing) continue;
        if (e.status == GameStatus::UpdateAvailable) { ++updates; only = e.name; ++ready; }
        else if (e.status == GameStatus::NeedsRepair) ++repairs;
        else if (e.status == GameStatus::Orphaned) ++gone;
        else if (e.status == GameStatus::Leftovers) ++left;
        else if (ToAdopt(e)) ++adopt;
        else if (NeedsTool(e)) { ++tool; toolName = e.needs; }
        else if (RrConfigReadyToAdd(e)) ++rrcfg;  // r3 #5
    }
    for (auto& e : v.games) readyAll += e.status == GameStatus::UpdateAvailable && !(v.busy && IEq(v.busyDir, e.dir));
    // the heading counts the group's games that still need attention (a game the running Update all has finished keeps
    // its place on the shelf, not its place in the count), the banner's game with them as the badge counts it; that one is
    // named in the caption, so the count matches what you see ("Needs attention 2" over one lockup: "Lantern Row is on the
    // banner above.")
    int still = 0;
    wstring onBanner;
    for (int i : group) {
        if (!Attention(v.games[i])) continue;
        ++still;
        if (std::find(listed.begin(), listed.end(), i) == listed.end()) onBanner = v.games[i].name;
    }
    const bool button = running || (ready > 0 && ready == readyAll);
    const wstring label = running ? wstring(L"Updating…") : L"Update " + std::to_wstring(ready) + (ready == 1 ? L" game" : L" games");
    const float bw = button ? std::ceil(gx.TextW(label, Ts(13, 600, pal.text)) + 32) : 0;
    ShelfHeading(gx, 0, x0, y, kShelfHeadH, cw - (button ? bw + 16 : 0), L"Needs attention", still, 20);
    if (button) {
        const bool on = !v.busy && !running && !v.listCached;
        // an update is routine: the accent's fill (the player's pick), never red, which is kept for what is broken (UI review #3)
        const float w = Button(gx, ID_UPDATEALL, x0 + cw, y, label, B_GO, 0, on, kShelfHeadH, bw, true);
        if (v.listCached && !v.busy && !running) { const wstring why = BusyWhy(GameEntry()); v.tips[ID_UPDATEALL] = why; Describe(ID_UPDATEALL, why); }
        if (on) slots.push_back({-1, Rc{x0 + cw - w, y, w, kShelfHeadH}, ID_UPDATEALL});  // Up from the first row of lockups
    }
    wstring lead;
    if (running) {
        wstring who;
        for (auto& e : v.games) if (v.busy && IEq(v.busyDir, e.dir)) who = e.name;
        lead = L"Updating " + (who.empty() ? wstring(L"your games") : who) + L" (" + std::to_wstring(std::min(v.queueDone + 1, v.queueTotal)) + L" of " +
               std::to_wstring(v.queueTotal) + L"). Each game keeps its settings.";  // the step is on the game's lockup
    } else {
        // "WrongGPU 0.5.5 is ready for 5 games, and 2 need a repair." / "1 game is gone, and 1 has leftover WrongGPU files."
        std::vector<wstring> said;
        const bool more = doing && doing->status == GameStatus::UpdateAvailable;
        if (updates) said.push_back(L"WrongGPU " + wstring(Ctx().data->version) + L" is ready for " +
                                    (updates == 1 ? only + (more ? L" too" : L"") : std::to_wstring(updates) + (more ? L" more games" : L" games")));
        // "games" once, in the first count (after "ready for 5 games" none needs it; after "ready for Hogwarts Legacy" the
        // next one does), and always with "is gone" ("1 game is gone": "1 is gone" alone reads as nothing)
        bool nounSaid = updates > 1;
        auto count = [&](int n, const wchar_t* one, const wchar_t* many, bool noun = false) {
            if (!n) return;
            said.push_back(std::to_wstring(n) + (!nounSaid || noun ? (n == 1 ? L" game " : L" games ") : L" ") + (n == 1 ? one : many));
            nounSaid = true;
        };
        count(repairs, L"needs a repair", L"need a repair");
        count(gone, L"is gone", L"are gone", true);
        count(left, L"has leftover WrongGPU files", L"have leftover WrongGPU files");
        count(adopt, L"was set up outside the launcher", L"were set up outside the launcher");
        const wstring needsTool = L"needs " + toolName + L" again", needTool = L"need " + toolName + L" again";
        count(tool, needsTool.c_str(), needTool.c_str());
        count(rrcfg, L"has its Ray Reconstruction setting ready to add", L"have their Ray Reconstruction setting ready to add");
        for (size_t k = 0; k < said.size(); ++k) lead += (k == 0 ? L"" : k + 1 == said.size() ? L", and " : L", ") + said[k];
        if (said.empty()) lead = doing ? wstring() : L"Open a game to see what it needs.";
        // the settings sentence is the updates': with other kinds in the list it says so ("Updates keep each game's settings.")
        else if (const int total = updates + repairs + gone + left + adopt + tool + rrcfg; updates)
            lead += total == 1 ? L". It keeps its settings." : updates == total ? L". Each game keeps its settings."
                    : updates == 1 ? L". The update keeps the game's settings." : L". Updates keep each game's settings.";
        else lead += repairs ? L". Repair puts WrongGPU's files back."
                     : gone + left ? L". Review shows what is left and how to clean it up."
                     : tool && !adopt ? (tool == 1 ? L". Its page says what to do." : L". Their pages say what to do.")
                     : adopt ? (adopt == 1 ? L". Adopt keeps its settings." : L". Adopt keeps their settings.")
                     // the Engine.ini line alone: what its pill does, not Adopt's sentence (r4 #8: it fell through to it)
                     : rrcfg == 1 ? L". Fix it now adds it. Windows asks for permission first." : L". Fix it now adds each. Windows asks for permission first.";
        if (doing) {
            wstring verb = v.busyVerb;
            while (!verb.empty() && (verb.back() == L'\u2026' || verb.back() == L'.')) verb.pop_back();
            if (verb.empty()) verb = L"Working on";
            lead = verb + L" " + doing->name + L"." + (lead.empty() ? L"" : L" " + lead);
        }
    }
    if (!onBanner.empty()) lead += (lead.empty() ? L"" : L" ") + onBanner + L" is on the banner above.";
    TS ls = Ts(13.5f, 400, pal.text2); ls.wrap = true; ls.lineH = 19;
    const float lh = gx.TextH(lead, cw, ls);
    gx.Text(lead, Rc{x0, y + kShelfHeadH + 3, cw, lh + 2}, ls);
    AddText(Rc{x0, y + kShelfHeadH + 3, cw, lh + 2}, lead);
    return kShelfHeadH + 3 + lh;
}

// While the first scan runs: one shelf of blank lockups where the games will be (the art's shape, a name, a line, the
// action), left on the column as the real shelves are. One after the other lights up (Apple HIG, Progress indicators:
// "Keep progress indicators moving"), every 420 ms, each change a 140 ms cross-fade drawn at the display's rate; between
// the changes nothing moves and nothing is drawn (timer 1 wakes the window for the next one). None lights up when
// Windows' Animation effects are off, or in a still frame. Returns the height.
constexpr ULONGLONG kSkelStepMs = 420, kSkelFadeMs = 140;
static float SkeletonShelf(Gfx& gx, float x0, float top) {
    const int perRow = grid.n;
    int lit = -1, was = -1;
    float in = 1;  // how far the step to `lit` has come (0..1)
    if (v.live && !gx.dry && MotionOn()) {
        const ULONGLONG now = NowMs(), step = now / kSkelStepMs, phase = now % kSkelStepMs;
        lit = (int)(step % (ULONGLONG)perRow);
        was = (int)((step + (ULONGLONG)perRow - 1) % (ULONGLONG)perRow);
        if (phase < kSkelFadeMs) {
            const float t = (float)phase / kSkelFadeMs;
            in = t * t * (3 - 2 * t);  // ease in and out
            libMoving = true;
        } else {
            SetTimer(v.hwnd, 1, (UINT)(kSkelStepMs - phase), nullptr);  // the next step (WM_TIMER 1 asks for a frame)
        }
    }
    static const float kLine[4] = {0.78f, 0.56f, 0.68f, 0.62f};
    const Col base = pal.fill, glow = Mix(pal.fill, pal.text, pal.dark ? 0.08f : 0.05f);
    gx.FillRound(Rc{x0, top + 9, 150, 12}, 6, base);  // the heading
    const float ty = top + kShelfHeadH + kTilesGap;
    const float aw = grid.artW, tw = aw - kLockPillMinW - 12;
    for (int k = 0; k < perRow; ++k) {
        const float x = std::round(x0 + k * (aw + grid.gap));
        const float lift = k == lit ? in : k == was && was != lit ? 1 - in : 0;
        gx.FillRound(Rc{x, ty, aw, grid.artH}, kArtR, Mix(base, glow, lift));
        gx.FillRound(Rc{x + 1, ty + MetaY() + 5, std::round(tw * kLine[k % 4]), 9}, 4.5f, base);
        gx.FillRound(Rc{x + 1, ty + LineY() + 4, std::round(tw * kLine[k % 4] * 0.6f), 8}, 4, base);
        gx.FillRound(Rc{x + aw - kLockPillMinW, ty + MetaY() + 1, kLockPillMinW, kLockPillH}, kLockPillH / 2, base);
    }
    return kShelfHeadH + kTilesGap + TileH();
}

// Rescan's glyph while a scan runs: the refresh arrows turning once a second, in the secondary text colour (it is busy, not
// off), on the display's clock as the skeleton's steps are (libMoving keeps the frames coming only while it shows). Still
// with Windows' Animation effects off and in a still frame: then the button's off colour, and the title line says it looks.
static void ScanGlyph(Gfx& gx, float cx, float cy) {
    const bool turns = v.live && !gx.dry && MotionOn();
    if (gx.dry || !gx.rt) return;
    if (!turns) { gx.Icon(ic::refresh, cx, cy, 13, pal.text3); return; }
    libMoving = true;
    const float deg = (float)(NowMs() % 1000) * 0.36f;
    D2D1_MATRIX_3X2_F old;
    gx.rt->GetTransform(&old);
    gx.rt->SetTransform(D2D1::Matrix3x2F::Rotation(deg, D2D1::Point2F(cx, cy)) * old);
    gx.Icon(ic::refresh, cx, cy, 13, pal.text2);
    gx.rt->SetTransform(old);
}

// The mark over a search with nothing in it: the search glyph on a round fill. Not the blank shelf: blank lockups where
// the games would be read as games still loading.
static void EmptyMark(Gfx& gx, float cx, float top, wchar_t glyph) {
    gx.FillRound(Rc{std::round(cx - 32), top, 64, 64}, 32, pal.fill);
    gx.Icon(glyph, cx, top + 33, 24, pal.text2);
}

// What the Library says under its title while every action on it waits (v.busy) and no lockup shows why: a job that is
// not a game's (RunTask: converting Model M, installing the launcher, collecting logs), or a game's operation
// whose lockup the search hides. Empty when a lockup or the running "Update N games" shows it.
static wstring BusyNote() {
    if (!v.busy || v.queueTotal > 0) return wstring();
    wstring doing;
    if (!v.busyDir.empty()) {
        const GameEntry* game = nullptr;
        for (const GameEntry& e : v.games) if (IEq(e.dir, v.busyDir)) game = &e;
        if (game && (v.search.empty() || Lower(game->name).find(Lower(v.search)) != wstring::npos)) return wstring();  // its lockup shows it
        wstring verb = v.busyVerb;
        while (!verb.empty() && (verb.back() == L'…' || verb.back() == L'.')) verb.pop_back();
        doing = (verb.empty() ? wstring(L"Working on") : verb) + (game ? L" " + game->name : wstring(L" a game")) + L"…";
    } else {
        switch (v.task) {
            case T_MODEL: doing = L"Converting the Model M data…"; break;
            case T_RRDATA: doing = L"Converting the Ray Reconstruction data\u2026"; break;
            case T_SELFINSTALL: doing = L"Installing WrongGPU for this user…"; break;  // the toast's and BusyWhy's words
            case T_SELFUNINSTALL: doing = L"Uninstalling the launcher…"; break;
            case T_FIRSTRUN:  // Install for this user, or (portable, no choice) the model and the adoptions only
                doing = v.first.installOffer && v.first.choice == 1 ? L"Installing WrongGPU for this user…" : L"Setting up WrongGPU…";
                break;
            case T_COLLECT: doing = L"Collecting logs…"; break;
            default: doing = L"Working…"; break;
        }
    }
    // what waits, as PlayBlocked has it: Play waits for a job that is not a game's, but not for collecting the logs (it only
    // reads) or for another game's operation (only that game's Play waits, and its lockup is hidden)
    const bool playWaits = v.task != T_COLLECT && (v.task != T_NONE || v.busyDir.empty());
    // (Repair waits in both: every operation does, one at a time; the lockups' Repair pills are off with it)
    return doing + (playWaits ? L" Install, Update, Repair and Play wait until it's done." : L" Install, Update and Repair wait until it's done.");
}

// The featured banner's game: the one played most recently with WrongGPU on; if none has been played, the first that needs a
// repair or an update (its action leads, in red); else none (a library without a game to continue or to fix: no banner).
// While Update all runs the banner keeps the game it showed when the queue began (one of the queue's: it is left out of
// "Needs attention", so the shelf would move under the pointer if the banner took the next game as each one finishes).
static wstring featuredDir;
static int FeaturedPick();
static int FeaturedGame() {
    if (v.queueTotal > 0 && !featuredDir.empty())
        for (int i = 0; i < (int)v.games.size() && i < kTileSpan; ++i)
            if (IEq(v.games[i].dir, featuredDir) && InQueueRun(v.games[i]) &&
                (v.games[i].status == GameStatus::Installed || v.games[i].status == GameStatus::UpdateAvailable)) return i;
    const int best = FeaturedPick();
    featuredDir = best >= 0 ? v.games[best].dir : wstring();
    return best;
}
static int FeaturedPick() {
    int best = -1;
    for (int i = 0; i < (int)v.games.size() && i < kTileSpan; ++i) {
        const GameEntry& e = v.games[i];
        if (!e.hasRecord || e.adoptable || !e.last.found || (e.status != GameStatus::Installed && e.status != GameStatus::Running)) continue;
        if (best < 0 || e.last.when > v.games[best].last.when) best = i;
    }
    if (best >= 0) return best;
    for (int i = 0; i < (int)v.games.size() && i < kTileSpan; ++i)
        if (v.games[i].status == GameStatus::NeedsRepair && (best < 0 || v.games[best].status != GameStatus::NeedsRepair)) best = i;
        else if (v.games[i].status == GameStatus::UpdateAvailable && best < 0) best = i;
    return best;
}

// The banner's button while its operation runs on its game ("Repair" -> "Repairing...", "Repair and update" -> "Repairing
// and updating..."), as the game page's button says it and as the banner's line over it does.
static const wchar_t* BannerBusyVerb(const wstring& act) {
    return act == L"Repair" ? L"Repairing…" : act == L"Repair and update" ? L"Repairing and updating…" : act == L"Update" ? L"Updating…" : L"";
}

// The featured banner (BE-wide .feature): the column's width, 240 to 300 high (BannerH), radius 16, a
// soft shadow. The game's hero at the banner's size (HeroCover, its focus 35 % from the top; the tile's while that
// decodes, else header.jpg, else its plate) under a scrim from the left: 0.74 -> 0.56 at 40 % (or past the words, whichever
// is further) -> 0 32 % later, so white words read 4.5:1 or better even on Cyberpunk's yellow. Bottom left: the logo (at
// most 300 x 96, less in a low banner; the name in 28 Display without one), one line ("Continue playing · Played Oct 17"),
// the status (the working green dot, "WrongGPU is on · Model M on Direct3D 12"), and two capsules: Play (the accent, the
// page's one filled button, 22 DIPs of padding as the mockup's .fplay) and Game settings (glass). A game that needs you:
// its Repair or Update in red and an amber status; while that runs, its button says so ("Repairing..."). Returns its height.
// The height (UI review #10): 240 at the default window (760 high: a full row of lockups under it), growing to 300 in a
// window 60 DIPs taller or more.
static float BannerH(float viewH) { return std::round(std::min(kBannerMaxH, std::max(kBannerMinH, viewH - 468))); }
float LibraryBannerH(float viewH) { return BannerH(viewH); }  // ui-dump (the launcher tests)
static float FeatureBanner(Gfx& gx, float x0, float y, float cw, float h, int index) {
    const GameEntry& e = v.games[index];
    Rc r{x0, y, cw, h};  // taller by the status's second line where the logo would shrink for it (below)
    const bool attention = e.status == GameStatus::NeedsRepair || e.status == GameStatus::UpdateAvailable;
    bool amber = false, on = false;
    wstring store;
    const wstring line = LockupLine(e, amber, on, store);  // "Played Oct 17", "Needs a repair", "0.5.0 -> 0.5.5"
    const wstring played = PlayedOn(e);
    // a game the running Update all has just updated, held on the banner until the queue ends (FeaturedGame): what it did.
    // While an operation runs on its game the line says so as its lockup would ("Updating... 62%", a progress bar after
    // it), and a game still to come in the running Update all says it waits ("Waiting to update · Not played yet")
    const bool busyHere = v.busy && IEq(v.busyDir, e.dir), waiting = !busyHere && Waiting(e);
    const wstring kicker = busyHere ? BusyLabel(e)
                           : waiting ? L"Waiting to update · " + played
                           : attention ? (e.status == GameStatus::NeedsRepair ? L"Needs a repair" : L"Update ready") + wstring(L" · ") + played
                           : RrConfigReadyToAdd(e) ? line + L" · " + played  // "Engine.ini: ready to add" (r3 #5)
                           : !e.last.found && InQueueRun(e) ? line + L" · " + played
                                                            : L"Continue playing · " + played;
    constexpr float kBarW = 160, kBarGap = 12;  // the busy line's progress bar, after its words
    wstring status, ran;  // ran: where it ran last (" on Direct3D 12"), what goes first when the line is too long
    Col dot = pal.green;
    bool update = false;  // an update ready: UpdateMark in the dot's place
    // "2 of WrongGPU's files are missing or changed. Repair puts them back." (the game page's count; WrongGPU named once)
    if (e.status == GameStatus::NeedsRepair) {
        const size_t n = e.problems.size();
        status = ProblemShort(e) + (!e.realExe.empty() ? L". Repair moves it to the game."
                                    : n == 1 ? L". Repair puts it back." : n > 1 ? L". Repair puts them back." : L". Repair puts WrongGPU's files back.");
        dot = pal.amber;
    }
    // an update ready is routine, neither a warning nor a failure: an up arrow in the line's colour, not a dot (UI review #3;
    // r2 #11: the accent's dot was the healthy green by default)
    else if (e.status == GameStatus::UpdateAvailable) { status = L"WrongGPU " + line + L". It keeps its settings."; update = true; }
    else {
        // the game page's header says the same ("Model M and Ray Reconstruction"): "on" is WrongGPU's state only, and
        // where it ran only with Ray Reconstruction off (its setup, not where it last ran: the page's Last run says that)
        status = L"WrongGPU is on · Model M";
        // the page header's rule (r6 #6): after a run of Ray Reconstruction alone, or one where RR did the upscaling, Model M had no
        // part in it: "Ray Reconstruction" alone
        if (RrAloneLastRun(e)) status = L"WrongGPU is on · Ray Reconstruction";
        else if (RrOn(e)) status += L" and Ray Reconstruction";  // a game that uses it, with WrongGPU's on (rr-notes.md)
        else if (e.last.ready && !e.last.executor.empty()) ran = e.last.executor == L"dx12" ? L" on Direct3D 12" : L" on HIP";
    }
    // the status in the words' room, no ellipsis while there is room for it (SHIP-BAR 2). The words keep to the scrim's
    // strong half (the left 50 %: a short line reads on any art there); where it ran goes first (the game's page and the
    // toolbar say it; UI Automation hears the whole line), then a second sentence, or the part after a middle dot, takes a
    // line of its own ("WrongGPU is on" / "Model M and Ray Reconstruction"), else the words wrap to a second line
    const float lx = x0 + 36, room = cw * 0.6f - 36, statusW = std::max(160.f, cw * 0.5f - 36 - 15);
    TS ss = Ts(14, 400, pal.onArt); ss.wrap = true; ss.lineH = 20; ss.trim = true;
    TS one = ss; one.wrap = false;
    wstring shown = status + ran;
    if (!ran.empty() && gx.TextW(shown, one) > statusW) shown = status;
    float statusH = 20;
    if (gx.TextW(shown, one) > statusW) {
        for (const wchar_t* cut : {L". ", L" · "}) {
            const size_t stop = shown.find(cut);
            if (stop == wstring::npos) continue;
            const bool midDot = cut[1] == L'·';
            const wstring head = shown.substr(0, midDot ? stop : stop + 1), tail = shown.substr(stop + (midDot ? 3 : 2));
            if (gx.TextW(head, one) <= statusW && gx.TextW(tail, one) <= statusW) { shown = head + L"\n" + tail; break; }
        }
        statusH = std::min(40.f, std::max(20.f, gx.TextH(shown, statusW, ss)));  // two lines hold every status
    }
    // the logo is the banner's only title: a status that wraps (a narrow window) takes its second line from the banner's
    // height, not from the logo, while the logo would be under its 96 DIPs (the stack under it: 30 + 36 + 18 + the status +
    // 3 + 18 + 16, and 18 over it at the least)
    if (const float logoRoom = r.h - (121 + statusH) - 18; statusH > 20 && logoRoom < 96) r.h += std::min(statusH - 20, 96 - logoRoom);
    gx.Shadow(r, kBannerR, 32, pal.dark ? 0.5f : 0.2f, 12, true);  // 0 1px 2px .12, 0 12px 32px .18 (the art covers its inner half)
    const Art a = e.appid.empty() ? Art{} : ArtFor(e.appid);
    float artIn = 1;  // the first hero fading in over the header or plate (its logo with it); never its dark bottom band
    HeroCover(gx, e.appid, r, kBannerR, 0.35f, [&] { if (a.header) gx.ImageCover(a.header, r, kBannerR, 0.5f); else Plate(gx, r, e, kBannerR, false); }, &artIn, kHeroKeep);
    // the buttons: what the tile's pill does (Play, Repair, Repair and update, Update), then the game's page. The one whose
    // operation runs on this game says its verb, disabled, as the game page's button does ("Repairing..."), without a
    // "waits" tooltip. Measured here: the scrim under them counts them as it counts the words
    bool tint = false;
    const wstring act = LockupAction(e, tint);
    const bool acts = act == L"Play" || act == L"Repair" || act == L"Repair and update" || act == L"Update" || act == L"Fix it now";  // (r3 #5)
    const wstring verb = acts ? BannerBusyVerb(act) : L"";
    const bool doing = !verb.empty() && busyHere && (v.busyVerb.empty() || v.busyVerb == verb);
    // a game still to come in the running Update all: "Waiting..." as its lockup says it (and its kicker), not "Update"
    const wstring label = doing ? verb : waiting ? wstring(L"Waiting…") : act;
    const BtnKind kind = act == L"Play" || act == L"Update" || act == L"Fix it now" ? B_GO : B_PRIMARY;  // red only for a repair (UI review #3)
    const wchar_t icon = act == L"Play" ? ic::playLine : 0;
    const float actW = acts ? ButtonW(gx, label, icon, 36, kind) + 8 : 0;
    const float buttonsR = (acts ? actW + 10 : 0) + ButtonW(gx, L"Game settings", 0, 36, B_GLASS);
    // the scrim from the left: 0.74 -> 0.56 -> 0, its 0.56 stop at 40 % of the banner or past the words' and the buttons'
    // right end, whichever is further, so white words read 4.95:1 or better on any art under them (white art under a 0.56
    // black: 4.95:1; Cyberpunk's yellow: 5.2:1), at every width (SHIP-BAR 2), and the glass button's label has the scrim's
    // 0.56 or more under it (FeatureBanner's glass, below). The stop is kept to whole hundredths: one brush per width, made
    // once (Gfx::GradStops caches it).
    {
        float wordsR = std::max(buttonsR, gx.TextW(kicker, Ts(13, 400, pal.onArt)) + (busyHere || waiting ? kBarGap + kBarW : 0));
        size_t from = 0;
        for (;;) {  // the status's widest line
            const size_t nl = shown.find(L'\n', from);
            wordsR = std::max(wordsR, 15 + std::min(statusW, gx.TextW(shown.substr(from, nl == wstring::npos ? wstring::npos : nl - from), one)));
            if (nl == wstring::npos) break;
            from = nl + 1;
        }
        if (!(a.logo && a.hero)) wordsR = std::max(wordsR, std::min(room, gx.TextW(e.name, Ts(28, 600, pal.onArt, F_DISPLAY))));
        const float strong = std::min(0.8f, std::max(0.40f, std::ceil((lx + wordsR + 16 - x0) / cw * 100) / 100));
        const float clear = std::min(1.f, strong + 0.32f);
        if (clear < 1) gx.GradStops(r, kBannerR, {{0.f, Alpha(pal.black, 0.74f)}, {strong, Alpha(pal.black, 0.56f)}, {clear, Alpha(pal.black, 0)}, {1.f, Alpha(pal.black, 0)}}, 0, 0.5f, 1, 0.5f);
        else gx.GradStops(r, kBannerR, {{0.f, Alpha(pal.black, 0.74f)}, {strong, Alpha(pal.black, 0.56f)}, {1.f, Alpha(pal.black, 0)}}, 0, 0.5f, 1, 0.5f);
    }
    if (pal.dark) gx.StrokeRound(r.in(-0.5f), kBannerR + 0.5f, Alpha(pal.white, 0.07f), 1);
    const ArtScope onArt;  // the status dot and the buttons' focus ring in the art's colours (both themes)
    // from the bottom: the buttons 30 up, the status 18 over them, the line 3 over that, the logo 16 over the line, and at
    // least 18 over the logo (a low banner takes the rest from the logo's height, after the status's second line took its
    // own from the banner's: the logo stays the banner's title at the smallest window too)
    const float by = r.b() - 30 - 36, sy = by - 18 - statusH, ky = sy - 3 - 18, logoB = ky - 16;
    const float logoH = std::max(40.f, std::min(96.f, logoB - r.y - 18));
    float headH = logoH;
    if (a.logo && a.hero) gx.ImageContain(a.logo, Rc{lx, logoB - logoH, std::min(300.f, room), logoH}, 0, 2, artIn, 10);
    else {
        TS ns = Ts(28, 600, pal.onArt, F_DISPLAY); ns.wrap = true; ns.lineH = 33; ns.trim = true;
        const float nh = std::min(logoH >= 66 ? 66.f : 33.f, gx.TextH(e.name, room, ns));
        gx.Text(e.name, Rc{lx, logoB - nh, room, nh + 2}, ns);
        headH = std::max(nh, 33.f);
    }
    // the logo is an image: its name, a heading
    AddHit(ID_FEAT0, Rc{lx, logoB - std::min(60.f, headH), std::min(300.f, room), std::min(60.f, headH)}, R_TEXT, e.name, true, 2, L"", false);
    TS ks = Ts(13, 400, Alpha(pal.onArt, 0.85f));
    gx.Text(kicker, Rc{lx, ky, room, 18}, ks);
    if (busyHere || waiting) {  // the lockup's progress (ProgressBar: the run red to green; waiting: its track), after the words
        const float bx0 = lx + std::ceil(gx.TextW(kicker, ks)) + kBarGap;
        if (bx0 + kBarW <= lx + room) {
            // its track in the art's colours (both themes: white at 24 % over the scrim), not the page's fill (light grey on
            // the dark scrim reads as a full bar)
            const Col fill = pal.fill;
            pal.fill = Alpha(pal.onArt, 0.24f);
            ProgressBar(gx, Rc{bx0, ky + 7, kBarW, 4}, v.progress, busyHere);
            pal.fill = fill;
            AddProgress(0, Rc{bx0, ky + 4, kBarW, 10}, kicker, busyHere && v.progress > 0 ? v.progress : -1.f);
        }
    }
    if (update) UpdateMark(gx, lx + 3.5f, sy + 10.5f, pal.onArt);
    else StatusDot(gx, lx + 3.5f, sy + 10.5f, dot);
    gx.Text(shown, Rc{lx + 15, sy, statusW, statusH + 1}, ss);
    // ("Updating… WrongGPU 0.5.1 → 0.5.5.": a kicker that ends in its ellipsis takes no period after it)
    wstring said = kicker + (!kicker.empty() && kicker.back() == L'…' ? L" " : L". ") + status + ran;
    // each middle dot ends a sentence: its parts start with a capital ("Needs a repair. Not played yet.", not ", Not")
    for (size_t at; (at = said.find(L" \u00b7 ")) != wstring::npos;) said.replace(at, 3, L". ");
    if (!said.empty() && said.back() != L'.') said += L'.';
    AddText(Rc{lx, ky, room, sy + statusH - ky}, said);
    float bx = lx;
    if (acts) {
        const bool en = act == L"Play" ? !PlayBlocked(e) : !v.busy && !v.listCached && e.status != GameStatus::Running;
        const float w = Button(gx, ID_FEAT0 + 1, bx, by, label, kind, icon, en, 36, actW, false,
                               waiting ? L"Waiting to update " + e.name : act == L"Fix it now" ? L"Fix Engine.ini now in " + e.name : label + L" " + e.name);
        if (!en && !doing) { const wstring why = BusyWhy(e); if (!why.empty()) { v.tips[ID_FEAT0 + 1] = why; Describe(ID_FEAT0 + 1, why); } }
        if (en) slots.push_back({-1, Rc{bx, by, w, 36}, ID_FEAT0 + 1});
        bx += w + 10;
    }
    // the glass (white at 18 %, 26 % under the pointer, 32 % pressed) over a black of its own: the art shows through the
    // glass, and its white label reads 4.5:1 or better on any art in every state (SHIP-BAR 2: 13.5 semibold is not large
    // text). Worst case, white art under the scrim's 0.56 (112): 0.30 more black and the glass's 18 % -> 110, 5.1:1; under
    // the pointer 0.40 and 26 % -> 116, 4.7:1; pressed 0.54 and 32 % -> 117, 4.6:1. Lighter in each state on any art that is
    // not white, so hover and press still show.
    {
        const int gid = ID_FEAT0 + 2;
        const float under = Down(gid) ? 0.54f : Hot(gid) ? 0.40f : 0.30f;
        gx.FillRound(Rc{bx, by, ButtonW(gx, L"Game settings", 0, 36, B_GLASS), 36}, 18, Alpha(pal.black, under));
    }
    const float w2 = Button(gx, ID_FEAT0 + 2, bx, by, L"Game settings", B_GLASS, 0, true, 36, 0, false, L"Game settings for " + e.name);
    slots.push_back({-1, Rc{bx, by, w2, 36}, ID_FEAT0 + 2});
    return r.h;
}

// ---- the last session's list (SHIP-BAR 3: content in under 1 s) -------------------------------------------------------
// The start-up scan takes seconds (the store libraries, each game's folder, the installed files hashed); until it lands
// the Library shows the list as the last session left it, from library-list.txt in the launcher's data folder: one line
// per game, its fields separated by tabs (a tab or a line end inside one becomes a space), what the Library, the banner and
// a game's page header show. The first line names the format and the payload version: a list written by another version
// is not shown (its updates would be wrong). Meanwhile the operations wait (Install, Update, Repair, Adopt, Uninstall,
// Clean up: their entries are not read yet, BusyWhy says so); Play and the games' pages work. A game that was running
// is listed as it was before it ran.
static const wchar_t kListHead[] = L"wronggpu-library-list 2";  // 2: the flags carry rrDirect and optiEdition (a list of 1 is not read)
constexpr size_t kListFields = 23;
static wstring ListPath() { return gl.paths.dataDir.empty() ? wstring() : Join(gl.paths.dataDir, L"library-list.txt"); }

void SaveLibrarySnapshot() {
    if (!v.live || v.fixture || v.listCached || !Ctx().data) return;  // only a list a scan read (never the one read back)
    const wstring path = ListPath();
    // only beside the settings (as SavePlacement): not before the first run saved them, not after the launcher uninstalled
    // itself (nothing of it is left behind)
    if (path.empty() || !IsFile(Join(gl.paths.dataDir, L"launcher.ini"))) return;
    auto field = [](wstring t) { for (wchar_t& c : t) if (c == L'\t' || c == L'\r' || c == L'\n' || c == 0x1f) c = L' '; return t; };
    auto bit = [](bool b) { return b ? L'1' : L'0'; };
    wstring out = wstring(kListHead) + L"\t" + field(Ctx().data->version) + L"\n";
    for (size_t i = 0; i < v.games.size() && i < (size_t)kTileSpan; ++i) {
        const GameEntry& e = v.games[i];
        wstring problems;
        for (const wstring& p : e.problems) problems += (problems.empty() ? L"" : wstring(1, (wchar_t)0x1f)) + field(p);
        const wstring flags = {bit(e.hasRecord), bit(e.recordV1), bit(e.perGameEntry), bit(e.hosted), bit(e.adoptable), bit(e.repairSuggested), bit(e.exeFound),
                               bit(e.hasPrevious), bit(e.rrUses), bit(e.rrInstalled), bit(e.rrEnabled), bit(e.rrPathTracingOnly), bit(e.needsFound),
                               bit(e.rrDirect), bit(e.optiEdition)};
        const wstring lastFlags = {bit(e.last.found), bit(e.last.ready), bit(e.last.optiPlugin), bit(e.last.upscaled), bit(e.last.rrOnly),
                                   bit(e.last.rrUpscaled)};  // (r4 #1, r6 #1)
        const wstring f[kListFields] = {e.name, e.exe, e.dir, e.source, e.appid, std::to_wstring((int)e.status), e.statusText, e.installedVersion, flags,
                                        e.previous, e.proxy, e.rrWhy, e.rrMode, e.needs, e.needsWhy, e.needsPage, e.needsGame, problems,
                                        e.last.when, e.last.upscaler, e.last.executor, e.last.process, lastFlags};
        for (size_t k = 0; k < kListFields; ++k) out += (k ? L"\t" : L"") + (k == 17 ? f[k] : field(f[k]));
        out += L"\n";
    }
    const string raw = ToUtf8(out);
    try { WriteFileAtomic(path, raw.data(), raw.size()); } catch (...) {}  // only a quicker start next time: nothing to say
}

bool LoadLibrarySnapshot() {
    if (!v.live || v.fixture || !Ctx().data) return false;
    const wstring path = ListPath();
    string raw;
    if (path.empty() || !ReadAll(path, raw)) return false;
    const wstring all = FromUtf8(raw);
    auto split = [](const wstring& t, wchar_t sep) {
        std::vector<wstring> parts;
        size_t from = 0;
        for (;;) {
            const size_t at = t.find(sep, from);
            parts.push_back(t.substr(from, at == wstring::npos ? wstring::npos : at - from));
            if (at == wstring::npos) return parts;
            from = at + 1;
        }
    };
    std::vector<wstring> lines = split(all, L'\n');
    if (lines.empty() || lines[0] != wstring(kListHead) + L"\t" + Ctx().data->version) return false;
    std::vector<GameEntry> games;
    for (size_t i = 1; i < lines.size() && games.size() < (size_t)kTileSpan; ++i) {
        const std::vector<wstring> f = split(lines[i], L'\t');
        if (f.size() != kListFields || f[0].empty() || f[8].size() != 15 || (f[22].size() != 3 && f[22].size() != 5 && f[22].size() != 6)) continue;  // (3: before r4; 5: before r6)
        GameEntry e;
        e.name = f[0]; e.exe = f[1]; e.dir = f[2]; e.source = f[3]; e.appid = f[4];
        const int st = _wtoi(f[5].c_str());
        if (st < (int)GameStatus::Eligible || st > (int)GameStatus::NeedsPrereq) continue;
        e.status = (GameStatus)st;
        e.statusText = f[6]; e.installedVersion = f[7];
        const wstring& b = f[8];
        e.hasRecord = b[0] == L'1'; e.recordV1 = b[1] == L'1'; e.perGameEntry = b[2] == L'1'; e.hosted = b[3] == L'1'; e.adoptable = b[4] == L'1';
        e.repairSuggested = b[5] == L'1'; e.exeFound = b[6] == L'1'; e.hasPrevious = b[7] == L'1'; e.rrUses = b[8] == L'1';
        e.rrInstalled = b[9] == L'1'; e.rrEnabled = b[10] == L'1'; e.rrPathTracingOnly = b[11] == L'1'; e.needsFound = b[12] == L'1';
        e.rrDirect = b[13] == L'1'; e.optiEdition = b[14] == L'1';
        e.previous = f[9]; e.proxy = f[10]; e.rrWhy = f[11]; e.rrMode = f[12];
        e.needs = f[13]; e.needsWhy = f[14]; e.needsPage = f[15]; e.needsGame = f[16];
        if (!f[17].empty()) e.problems = split(f[17], (wchar_t)0x1f);
        e.last.when = f[18]; e.last.upscaler = f[19]; e.last.executor = f[20]; e.last.process = f[21];
        e.last.found = f[22][0] == L'1'; e.last.ready = f[22][1] == L'1'; e.last.optiPlugin = f[22][2] == L'1';
        if (f[22].size() >= 5) { e.last.upscaled = f[22][3] == L'1'; e.last.rrOnly = f[22][4] == L'1'; }
        if (f[22].size() >= 6) e.last.rrUpscaled = f[22][5] == L'1';
        if (e.status == GameStatus::Running) {  // not running any more, most likely: as it was before it ran
            e.status = !e.hasRecord ? GameStatus::Eligible : !e.problems.empty() ? GameStatus::NeedsRepair : GameStatus::Installed;
            e.statusText.clear();
        }
        games.push_back(std::move(e));
    }
    if (games.empty()) return false;
    v.games = std::move(games);
    v.listCached = true;
    return true;
}

// ---- scroll anchoring (SHIP-BAR 3: nothing moves under the pointer) -----------------------------------------------------
// The lockup under the pointer, else (scrolled down) the first lockup in view: its game, its shelf and where its art was
// (content DIPs) in the last frame drawn. When the list changes above it (the start-up scan's full list adding "Needs
// attention" over "With WrongGPU", a game leaving a shelf above after an operation, a scan finding a new game), the page
// scrolls by as much as the lockup moved, so it stays where it was on the screen and a click lands on what was under the
// pointer. A resize that reflows the shelves (4 -> 3 -> 2 lockups a row, the banner lower) keeps it in place too, as a
// collection view keeps its first visible item. The game the player acted on (under the pointer, or busy in view) is
// followed to its new shelf ("Needs attention" -> "With WrongGPU" once its update is done, the shelf it left closing
// above): it stays where it was on the screen. Any other anchor holds only while its game keeps its shelf (else the page
// reflows as it would); never for a search, Show more / Show fewer or a page change (the layout is meant to change then).
struct Anchor {
    wstring key, search;
    int shelf = -1;
    float y = 0, w = 0, h = 0, scroll = 0;
    bool attnAll = false, follow = false;
    ULONGLONG pageAt = 0;
};
static Anchor anchor;
static unsigned long long listSig = 0;  // the list as the last frame drawn had it (LibrarySig)
static wstring AnchorKey(const GameEntry& e) { return Lower(e.dir) + L"|" + Lower(e.exe); }
// What the Library's layout follows, hashed (each frame: a few characters per game): the games, their places and states,
// the running queue and operation, the first scan. A frame whose hash differs is laid out once more, dry, to anchor.
static unsigned long long LibrarySig() {
    unsigned long long h = 1469598103934665603ull;
    auto mix = [&h](unsigned long long x) { h = (h ^ x) * 1099511628211ull; };
    auto text = [&mix](const wstring& t) { for (wchar_t c : t) mix((unsigned long long)towlower(c)); mix(0x1f); };
    mix(v.games.size());
    for (const GameEntry& e : v.games) {
        text(e.dir); text(e.exe); text(e.name); text(e.last.when);
        mix((unsigned long long)e.status * 8 + (e.hasRecord ? 4 : 0) + (e.adoptable ? 2 : 0) + (e.last.found ? 1 : 0));
    }
    mix((unsigned long long)v.queueTotal); mix((unsigned long long)v.queueDone); mix(v.scanning); mix(v.busy); text(v.busyDir);
    for (const wstring& d : v.queueDirs) text(d);
    return h;
}

void PaintLibrary(Gfx& gx, const Rc& area) {
    slots.clear();
    if (!gx.dry) libMoving = false;
    if (!gx.dry) {
        const unsigned long long sig = LibrarySig();
        const bool resized = anchor.w != area.w || anchor.h != area.h;
        if (v.live && (sig != listSig || resized) && !anchor.key.empty() && anchor.search == v.search && anchor.attnAll == v.attnAll &&
            anchor.pageAt == v.pageChangedAt) {
            Measure(gx, [&] { PaintLibrary(gx, area); return 0.f; });  // where the anchored lockup is now (slots), nothing drawn
            float shift = 0;
            bool found = false;
            for (const Slot& sl : slots)
                if (!sl.id && sl.index >= 0 && sl.index < (int)v.games.size() && (sl.shelf == anchor.shelf || anchor.follow) &&
                    AnchorKey(v.games[(size_t)sl.index]) == anchor.key) {
                    shift = sl.r.y - anchor.y;
                    found = true;
                    break;
                }
            slots.clear();
            // a resize: from the scroll the anchor was seen at (PaintAll clamped it to the new view with the old page's height)
            float& sc = v.scroll[P_LIBRARY];
            const float base = resized ? anchor.scroll : sc;
            if (found && std::fabs(base + shift - sc) > 0.5f) {  // the scroll, its target and this frame's offset by as much (within the page)
                float& tg = v.target[P_LIBRARY];
                const float maxScroll = std::max(0.f, v.contentH[P_LIBRARY] - v.H);
                const float to = std::min(std::max(0.f, base + shift), maxScroll), moved = to - sc;
                sc = to;
                tg = std::min(std::max(0.f, tg + moved), maxScroll);
                const float was = cv.scroll;
                cv.scroll = std::round(sc * gx.scale) / gx.scale;
                gx.SetOffset(0, gx.OffsetY() - (cv.scroll - was));
            }
        }
        listSig = sig;
    }
    // the column every page shares: its left and right edges never move between pages. The tools, Update N games, the
    // captions, the hairlines (BE-signal's .wall) and the rows of lockups all span it: one right edge (GridFor)
    const float x0 = ColX(area);
    const float cw = ColW(area.w);
    grid = GridFor(cw);
    const int perRow = grid.n;
    v.pageCx = x0 + cw / 2;
    const float top = area.y + 18;
    // ---- the head: the large title and its line; Search, Rescan (an icon) and Add game at the right, on the line's bottom
    std::set<wstring> stores;
    for (auto& e : v.games) stores.insert(StoreOf(e));
    // "12 games on Steam and Epic", "14 games on Steam, Epic, GOG and in your folders"
    std::vector<wstring> parts;
    for (const wchar_t* s : {L"Steam", L"Epic", L"GOG"}) if (stores.count(s)) parts.push_back(s);
    const bool others = stores.count(L"Your folder") || stores.count(L"Added by you");
    wstring where;
    for (size_t i = 0; i < parts.size(); ++i) where += (i == 0 ? L"" : i + 1 == parts.size() && !others ? L" and " : L", ") + parts[i];
    if (others) where = parts.empty() ? L" in your folders" : L" on " + where + L" and in your folders";
    else if (!where.empty()) where = L" on " + where;
    // a scan over the list shown (the last session's, or Rescan): the scan's one name, as Rescan's tooltip, the empty
    // Library and Narrator say it
    const wstring looking = v.scanning ? L" · Looking for games…" : L"";
    const wstring allSub = v.scanning && v.games.empty() ? L"Looking for games…"
                           : v.games.empty() ? L"No games found yet"
                           : std::to_wstring(v.games.size()) + (v.games.size() == 1 ? L" game" : L" games") + where + looking;
    // during a search it counts the results, as the shelves' counts do: "1 of 21 games"
    wstring sub = allSub;
    if (!v.search.empty() && !v.games.empty()) {
        const wstring find = Lower(v.search);
        size_t match = 0;
        for (const GameEntry& e : v.games) match += Lower(e.name).find(find) != wstring::npos;
        sub = std::to_wstring(match) + L" of " + std::to_wstring(v.games.size()) + (v.games.size() == 1 ? L" game" : L" games") + looking;
    }
    TS h1 = Ts(34, 600, pal.text, F_DISPLAY); h1.spacing = -0.4f;
    const TS ss = Ts(14, 400, pal.text2);
    const bool search = !v.games.empty();  // no games at all: no search (the empty state says what to do)
    const float addW = ButtonW(gx, L"Add game…", ic::add, 32, B_GHOST);  // it opens a file picker: an ellipsis, as Add folder... has
    float sw = search ? (v.W < 1100 ? 180.f : 240.f) : 0;
    const float toolsW = (search ? sw + 8 : 0) + 32 + 8 + addW;
    // (the whole count's width during a search too: the tools do not move as you type)
    const float textW = std::max({gx.TextW(L"Library", h1), gx.TextW(sub, ss), gx.TextW(allSub, ss)}) + 1;
    const bool beside = textW + 24 + toolsW <= cw;  // a narrow window: the tools on a row of their own under the title
    const float textRoom = beside ? cw - toolsW - 24 : cw;
    PageTitle(gx, L"Library", Rc{x0, top - 4, textRoom, 44}, h1);  // the font's line is taller than the mockup's 1.15
    // a job that is not a game's (converting Model M, installing the launcher, collecting logs), or a game's operation
    // whose lockup the search hides: every action of the Library waits for it, and the title's line says why in the
    // count's place while it runs. Its own fixed line: nothing under it moves when the job starts or ends, or when a
    // search hides or shows the busy game. One line: the job, then what waits when that fits (the whole note in its
    // tooltip and for UI Automation when it is cut)
    const wstring note = BusyNote();
    if (!note.empty()) {
        wstring line = note;
        if (gx.TextW(line, ss) > textRoom) {
            const size_t stop = note.find(L"… ");  // "Converting the Model M data…" without "Install, Update and Play wait..."
            if (stop != wstring::npos) line = note.substr(0, stop + 1);
        }
        gx.Text(line, Rc{x0, top + 43, textRoom, 20}, ss);
        const int tid = AddText(Rc{x0, top + 43, std::min(textRoom, gx.TextW(line, ss) + 1), 20}, note);
        if (tid && (line != note || gx.TextW(line, ss) > textRoom)) v.tips[tid] = note;
    } else {
        gx.Text(sub, Rc{x0, top + 43, textRoom, 20}, ss);
        AddText(Rc{x0, top + 43, std::min(textRoom, gx.TextW(sub, ss) + 1), 20}, sub);  // Narrator reads it after the title
    }
    float y = top + 63;
    float tx = x0 + cw - toolsW, ty = y - 4 - 32;
    if (!beside) {
        ty = y + 14;
        y = ty + 32;
        if (search) { sw = std::max(120.f, cw - (32 + 8 + addW) - 8); tx = x0; }
    }
    if (search) {
        FieldBox(gx, v.edSearch, ID_SEARCH, Rc{tx, ty, sw, 32}, v.search, L"Search games", ic::search, L"Search games", false, ID_SEARCHCLEAR);
        tx += sw + 8;
    }
    // while it looks: off (no hover, no hand, UI Automation says it is not enabled), but a keyboard focus on it stays there
    // with its busy name, as every control that goes busy keeps it (PaintAll), so pressing it from the keyboard never
    // throws the focus to Add game
    // while it looks its glyph turns (ScanGlyph): the button's own busy state, beside the title line's "Looking for games..."
    Button(gx, ID_RESCAN, tx, ty, L"", B_GHOST, v.scanning ? 0 : ic::refresh, !v.scanning, 32, 0, false, v.scanning ? L"Looking for games…" : L"Rescan for games");
    if (v.scanning) ScanGlyph(gx, tx + 16, ty + 16);
    v.tips[ID_RESCAN] = v.scanning ? L"Looking for games…" : L"Rescan for games";
    Button(gx, ID_ADDGAME, tx + 32 + 8, ty, L"Add game…", B_GHOST, ic::add, true, 32, addW);
    // ---- the featured banner (not during a search: the shelves are what it filters; not while the first scan runs)
    const wstring q = Lower(v.search);
    const int feat = q.empty() ? FeaturedGame() : -1;
    v.featIndex = feat;
    if (feat >= 0) { y += 26; y += FeatureBanner(gx, x0, y, cw, BannerH(area.h), feat); }
    // ---- the shelves: the games that need attention first (the most urgent first; the games the running queue has updated
    // keep their place until it ends), then the games with WrongGPU, the games it can go into, and those it stays out of.
    // The featured game is not repeated as a tile on any shelf (the banner has it and its action; no shelf when that leaves
    // it empty); "Needs attention" still counts it in its line and its button ("Update 5 games" updates it too), not in its
    // heading (that counts the lockups under it)
    std::vector<int> sec[4];
    // each shelf's games whether the search or the banner shows them: whether its tiles say where each game comes from
    // (ShelfStores) does not change with a search, the banner or Show more (a tile's line keeps its words as you type)
    std::vector<int> whole[4];
    for (int i = 0; i < (int)v.games.size() && i < kTileSpan; ++i) {  // the games' ids have room for kTileSpan of them
        const GameEntry& e = v.games[i];
        const int s = Attention(e) || InQueueRun(e) ? 0 : e.hasRecord ? 1 : Available(e) ? 2 : 3;
        whole[s].push_back(i);
        if (!q.empty() && Lower(e.name).find(q) == wstring::npos) continue;
        if (s == 1 && i == feat) continue;
        sec[s].push_back(i);
    }
    const std::vector<int> attnGroup = sec[0];  // AttentionHead's counts
    if (feat >= 0) sec[0].erase(std::remove(sec[0].begin(), sec[0].end(), feat), sec[0].end());
    // names two games share (a folder's copy of a Steam game): their lockups say where each comes from
    std::map<wstring, int> named;
    std::vector<wstring> lowName;  // the games' names in lower case (the shelves' order below), once a frame
    for (int i = 0; i < (int)v.games.size() && i < kTileSpan; ++i) { lowName.push_back(Lower(v.games[i].name)); ++named[lowName.back()]; }
    // each shelf in the scan's order (by name, SortGames), whatever order the list has meanwhile: a game installed in this
    // session (still where the scan put it among the games without WrongGPU) takes its place by name on "With WrongGPU",
    // as it has it after a Rescan, so the shelf's order never changes on a Rescan. "Needs attention" by urgency first.
    const auto byName = [&lowName](int a, int b) { return lowName[(size_t)a] < lowName[(size_t)b]; };
    for (int s = 1; s < 4; ++s) std::stable_sort(sec[s].begin(), sec[s].end(), byName);
    std::stable_sort(sec[0].begin(), sec[0].end(), [&](int a, int b) {
        const int ua = Urgency(v.games[a]), ub = Urgency(v.games[b]);
        return ua != ub ? ua < ub : byName(a, b);
    });
    const float seeTop = cv.scroll + cv.view.y, seeBottom = cv.scroll + cv.view.b();
    static const wchar_t* const titles[3] = {L"Needs attention", L"With WrongGPU", L"Available"};
    bool any = false;
    const bool skeleton = v.scanning && v.games.empty();  // the first scan: blank lockups where the games will be
    if (skeleton) { y += kShelfGap; y += SkeletonShelf(gx, x0, y); }
    for (int s = 0; s < 3; ++s) {
        // the update queue's progress stays in view while it runs, also when its last game has left the shelf
        if (sec[s].empty() && !(s == 0 && v.queueTotal > 0)) continue;
        y += kShelfGap;
        if (any) { Hairline(gx, x0, y, x0 + cw); y += 1 + kRuleGap; }
        any = true;
        if (s == 0) y += AttentionHead(gx, attnGroup, sec[0], x0, y, cw);
        else { ShelfHeading(gx, s, x0, y, kShelfHeadH, cw, titles[s], (int)sec[s].size(), 20); y += kShelfHeadH; }
        y += kTilesGap;
        // "Needs attention" shows two rows and "Show 12 more"; all of it during a search, while the queue runs or once
        // opened. Never "Show 1 more": a shelf one lockup longer shows it
        const int n = (int)sec[s].size(), few = kAttnRows * perRow;
        const bool folds = s == 0 && q.empty() && v.queueTotal == 0 && n > few + 1;
        const int shown = folds && !v.attnAll ? few : n;
        // row by row; where the games come from on every tile of the shelf or on none
        const bool withStores = ShelfStores(gx, whole[s], named);
        const float rowH = TileH();
        for (int r0 = 0; r0 < shown; r0 += perRow) {
            const int r1 = std::min(shown, r0 + perRow);
            if (r0) y += kLockRowGap;
            for (int k = r0; k < r1; ++k) {
                const GameEntry& e = v.games[sec[s][k]];
                const float lx = std::round(x0 + (float)(k - r0) * (grid.artW + grid.gap));
                // only the tiles in view are drawn (and ask for their art): a library of hundreds of games stays smooth
                Lockup(gx, sec[s][k], lx, y, MeasureLockup(gx, e, named[Lower(e.name)] > 1, withStores), !(y + rowH + 30 < seeTop || y - 30 > seeBottom));
                if (!slots.empty()) slots.back().shelf = s;
            }
            y += rowH;
        }
        if (folds) {  // under the shelf it adds to: "Show 12 more" and the first of their names; once open, "Show fewer"
            y += 16;
            const bool open = v.attnAll;
            const std::vector<int> hidden(sec[s].begin() + few, sec[s].end());
            const wstring lead = open ? wstring(L"Show fewer") : L"Show " + std::to_wstring(hidden.size()) + L" more";
            wstring all;
            for (size_t k = 0; k < hidden.size(); ++k) all += (k ? L", " : L"") + v.games[hidden[k]].name;
            // the button's fill starts on the column's edge, where the lockups start
            const float bw = Button(gx, ID_ATTNMORE, x0, y, lead, B_GHOST, open ? ic::up : ic::down, true, 32, 0, false,
                                    open ? wstring(L"Show fewer games that need attention")
                                         : lead + (hidden.size() == 1 ? L" game that needs attention: " : L" games that need attention: ") + all);
            slots.push_back({-1, Rc{x0, y, bw, 32}, ID_ATTNMORE});  // Down from the shelf's last row
            const float nx = x0 + bw + 12;
            if (!open) {
                // as many whole names as fit, then how many others: "Borderlands 4, The Witcher 3 and 26 others"
                TS ns = Ts(13, 400, pal.text2); ns.valign = 1;
                wstring names;
                for (size_t k = hidden.size(); k-- > 0;) {
                    wstring t;
                    for (size_t j = 0; j <= k; ++j) t += (j ? L", " : L"") + v.games[hidden[j]].name;
                    const size_t rest = hidden.size() - k - 1;
                    if (rest) t += L" and " + std::to_wstring(rest) + (rest == 1 ? L" other" : L" others");
                    if (gx.TextW(t, ns) <= x0 + cw - nx) { names = t; break; }
                }
                if (!names.empty()) gx.Text(names, Rc{nx, y, x0 + cw - nx, 32}, ns);
            }
            y += 32;
        }
    }
    if (!sec[3].empty()) {  // the games WrongGPU stays out of: compact rows after a hairline
        y += kShelfGap;
        if (any) { Hairline(gx, x0, y, x0 + cw); y += 1 + 26; }
        any = true;
        ShelfHeading(gx, 3, x0, y, 18, cw, L"Not supported", (int)sec[3].size(), 15);  // the game page's word for it too
        y += 18;
        for (int i : sec[3]) {
            y += 12;
            RefusedRow(gx, i, x0, y, cw, !(y + kRefRowH + 30 < seeTop || y - 30 > seeBottom));
            if (!slots.empty()) slots.back().shelf = 3;
            y += kRefRowH;
        }
    }
    v.rackEmpty = !any;
    if (!any && !skeleton) {  // empty states: what happened and what to do next, under a mark
        const bool none = v.games.empty();
        wstring title, msg;
        if (none) {
            title = L"Where are your games?";
            msg = L"WrongGPU looked in your Steam, Epic and GOG libraries and found none. A game somewhere else? Add the game's program, or add the folder that holds it.";
        }
        else { title = L"No game matches “" + v.search + L"”"; msg = L"Check the spelling, or clear the search to see all your games."; }
        TS ts = Ts(20, 600, pal.text, F_DISPLAY); ts.align = 1; ts.wrap = true;
        TS ms = Ts(13.5f, 400, pal.text2); ms.align = 1; ms.wrap = true; ms.lineH = 19;
        const float mw = std::min(cw - 48, 440.f), mh = gx.TextH(msg, mw, ms), th = gx.TextH(title, cw - 48, ts);
        std::vector<std::pair<int, wstring>> acts;
        if (!v.scanning && none) acts = {{ID_E_ADDGAME, L"Add game…"}, {ID_E_SETTINGS, L"Add folder\u2026"}};  // Settings' name for it; it opens the folder picker
        else if (!v.scanning && !v.search.empty()) acts = {{ID_E_CLEAR, L"Clear search"}};
        const float markH = 64.f;
        const float blockH = markH + 24 + th + 6 + mh + (acts.empty() ? 0 : 20 + 34);
        // nothing on the page at all: in the middle of the page; a search without a match: where the games would be
        const float etop = none ? std::max(y + 24, std::floor(area.y + (area.h - blockH) / 2)) : y + 40;
        const float cx = x0 + cw / 2;
        EmptyMark(gx, cx, etop, none ? ic::grid : ic::search);
        float ey = etop + markH + 24;
        gx.Text(title, Rc{x0 + 24, ey, cw - 48, th + 2}, ts);
        AddText(Rc{x0 + 24, ey, cw - 48, th + 2}, title, 0, 2);  // "No game matches ..." for UI Automation, and what to do
        ey += th + 6;
        gx.Text(msg, Rc{std::round(cx - mw / 2), ey, mw, mh + 2}, ms);
        AddText(Rc{std::round(cx - mw / 2), ey, mw, mh + 2}, msg);
        ey += mh + 20;
        float bw = 0;
        for (auto& a : acts) bw += ButtonW(gx, a.second, 0, 34) + 10;
        float ex = std::round(cx - (bw - 10) / 2);
        for (size_t i = 0; i < acts.size(); ++i)
            ex += Button(gx, acts[i].first, ex, ey, acts[i].second, i == 0 && none ? B_GO : B_SECONDARY, 0, true, 34) + 10;
        y = etop + blockH;
    }
    v.contentH[P_LIBRARY] = y + 72;
    if (!gx.dry && v.live) {  // the next frame's anchor (a window on screen): the lockup under the pointer, else (scrolled
        // down) the busy game's lockup in view, else the first one in view; the first two are followed to another shelf
        anchor = Anchor{};
        const int hotIndex = v.hot >= ID_TILE && v.hot < ID_TILE + kTileSpan ? v.hot - ID_TILE : v.hot >= ID_TILEQA && v.hot < ID_TILEQA + kTileSpan ? v.hot - ID_TILEQA : -1;
        const Slot* pick = nullptr;
        bool follow = false;
        for (const Slot& sl : slots) if (!sl.id && sl.index >= 0 && sl.index == hotIndex && sl.shelf >= 0) { pick = &sl; follow = true; }
        if (!pick && cv.scroll > 0 && v.busy && !v.busyDir.empty())
            for (const Slot& sl : slots)
                if (!sl.id && sl.index >= 0 && sl.index < (int)v.games.size() && sl.shelf >= 0 && sl.r.y >= seeTop && sl.r.b() <= seeBottom &&
                    IEq(v.games[(size_t)sl.index].dir, v.busyDir)) { pick = &sl; follow = true; break; }
        if (!pick && cv.scroll > 0)
            for (const Slot& sl : slots) if (!sl.id && sl.index >= 0 && sl.shelf >= 0 && sl.r.y >= seeTop) { pick = &sl; break; }
        if (pick && pick->index < (int)v.games.size()) {
            anchor.key = AnchorKey(v.games[(size_t)pick->index]);
            anchor.shelf = pick->shelf;
            anchor.follow = follow;
            anchor.y = pick->r.y;
            anchor.h = area.h;
            anchor.scroll = cv.scroll;
            anchor.w = area.w;
            anchor.search = v.search;
            anchor.attnAll = v.attnAll;
            anchor.pageAt = v.pageChangedAt;
        }
    }
}

// The arrow keys on the Library's lockups (launcher_ui.cpp's OnKey): Left and Right go to the previous and next lockup in
// reading order, Up and Down to the nearest lockup of the row above or below (the compact rows are rows of one). The
// shelves' buttons are rows of their own: "Update 4 games" above the first row of "Needs attention", "Show 12 more" /
// "Show fewer" under its last. A lockup over its action is one column (OnKey: Down from a lockup to its action, Up back),
// so Left and Right keep the stop (from an action to the target's action), Down lands on a lockup and Up on an action
// (the caller falls back to the lockup when it has none or it is disabled). Returns the id to focus, -1 with nothing that
// way (Left and Right do nothing, Up and Down scroll the page), 0 when the focus is not on a lockup or those buttons.
int LibraryArrow(int focus, WPARAM key) {
    if (v.page != P_LIBRARY) return 0;
    const bool act = focus >= ID_TILEQA && focus < ID_TILEQA + kTileSpan, tile = focus >= ID_TILE && focus < ID_TILE + kTileSpan;
    const int index = act ? focus - ID_TILEQA : tile ? focus - ID_TILE : -1;
    int at = -1;
    for (size_t i = 0; i < slots.size(); ++i)
        if (act || tile ? !slots[i].id && slots[i].index == index : slots[i].id == focus) at = (int)i;
    if (at < 0) return 0;
    int to = -1;
    if (key == VK_LEFT || key == VK_RIGHT) {
        to = key == VK_LEFT ? at - 1 : at + 1;
        if (to < 0 || to >= (int)slots.size()) return -1;
    } else if (key == VK_UP || key == VK_DOWN) {
        const Rc& c = slots[(size_t)at].r;
        const bool down = key == VK_DOWN;
        float rowY = 0, best = 1e9f;
        bool found = false;
        for (const Slot& s : slots) {  // the nearest row that way
            if (down ? s.r.y > c.y + 1 : s.r.y < c.y - 1) {
                if (!found || (down ? s.r.y < rowY : s.r.y > rowY)) rowY = s.r.y;
                found = true;
            }
        }
        if (!found) return -1;
        for (size_t i = 0; i < slots.size(); ++i)  // in it, the lockup nearest to this one's left edge
            if (std::fabs(slots[i].r.y - rowY) <= 1 && std::fabs(slots[i].r.x - c.x) < best) { best = std::fabs(slots[i].r.x - c.x); to = (int)i; }
        if (to < 0) return -1;
    } else return 0;
    const Slot& t = slots[(size_t)to];
    const bool toAct = key == VK_UP || ((key == VK_LEFT || key == VK_RIGHT) && act);
    return t.id ? t.id : (toAct ? ID_TILEQA : ID_TILE) + t.index;
}

}  // namespace wg::app::lui
