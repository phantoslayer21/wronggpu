// SPDX-License-Identifier: MIT
// WrongGPU launcher window: the first run's checks, on a worker (launcher_ui.cpp's StartFirstRun starts them; the sheet is
// ui_settings.cpp's). They only read: nothing on this PC changes before the player chooses and presses Continue. Written
// for this project.
#include "app/launcher_window.h"

#include <tlhelp32.h>

namespace wg::app {
using namespace wg::app::lui;

namespace win {

void FirstRunChecks(HWND hwnd, const GlobalSettings& gs, const wstring& dll, bool dump) {
    auto* m = new FirstRunMsg;
    // the lines so far, while the rest runs (the sheet fills its rows in as they finish instead of all at the end)
    auto partial = [&] {
        auto* p = new FirstRunMsg;
        p->lines = m->lines;
        if (!PostMessageW(hwnd, WM_APP_FIRSTRUN, 1, (LPARAM)p)) delete p;
    };
    // a throw anywhere in the checks (out of memory reading a broken file, a SetupError from a game) ends them with a
    // line that says so, never with the sheet waiting for good: what was found is posted as the stack unwinds (the
    // Worker logs the throw); Continue works with it
    struct PostAnyway {
        HWND hwnd; FirstRunMsg*& m;
        ~PostAnyway() {
            if (!m) return;
            try { m->lines.push_back({2, L"Checks: Didn't finish. Close WrongGPU and open it again to check again."}); } catch (...) {}
            if (!PostMessageW(hwnd, WM_APP_FIRSTRUN, 0, (LPARAM)m)) delete m;
        }
    } postAnyway{hwnd, m};
    const WinInfo win = WindowsInfo();
    m->lines.push_back({win.ok ? 0 : 2, L"Windows: " + win.text});
    // started inside an app package (MSIX), said in words a player knows: what happens, then what to do
    if (RunningPackaged())
        m->lines.push_back({2, L"Opened from: Another app. Games may not see the files WrongGPU writes while it's opened this way. Open WrongGPU from File Explorer instead."});
    const GpuInfo gpu = DetectGpu();
    // the card, then what it means in its own sentence ("AMD Radeon RX 6800 XT. WrongGPU needs ..."); the advice is said once, the
    // series kept on one line
    // release 0.6.0 (owner 2026-11-02): the RX 9060 series is supported but not yet tested on hardware; RR takes about twice
    // the RX 9070 XT's time there (an estimate), and its 8 GB cards are untested
    const bool rx9060 = gpu.rdna4 && gpu.name.find(L"9060") != wstring::npos;
    m->lines.push_back({gpu.rdna4 ? 0 : 2, L"Graphics card: " + (gpu.name.empty() ? wstring(L"Not found") : gpu.name) +
                                               (!gpu.rdna4 ? L". WrongGPU needs a Radeon\u00a0RX\u00a09000-series card."
                                                : rx9060   ? L". Supported, not yet tested on this card. Ray Reconstruction takes about twice as long as on an RX\u00a09070\u00a0XT."
                                                           : L"")});
    // short words behind the sheet's Show details (how Model M runs is the game page's Runs on); without the card, only
    // what it means here (the card's line above says what to do)
    m->lines.push_back({gpu.rdna4 ? 0 : 1, gpu.rdna4 ? L"Direct3D 12: Ready" : L"Direct3D 12: Not available on this card"});
    if (dump) m->lines.push_back({1, L"HIP runtime: Not checked in this run"});
    else {  // optional: a note, not a problem (the probe's own words go to setup.log)
        const HipInfo h = CheckHip(g.ctx);
        if (!h.ok) LogLine(g.ctx, L"first run: " + h.text);
        // found or not, and that it is optional (the card is the line above); the file, device and architecture go to the log
        if (h.ok) LogLine(g.ctx, L"first run: HIP runtime " + h.dll + L" works on " + h.device + L" (" + h.arch + L")");
        m->lines.push_back({h.ok ? 0 : 1, h.ok ? wstring(L"HIP runtime: Found (optional)")
                                               : wstring(L"HIP runtime: Not found (optional). Only Sparkle guard and the HIP choice in Runs\u00a0on need it.")});
    }
    partial();
    // the games first: the model's source game is named as the Library names it (the same list becomes v.games), the
    // store libraries only for a folder the Library does not list
    {
        std::shared_lock<std::shared_mutex> opti(optiLock);
        m->games = ScanGames(g.ctx, gs, g.noScan);
    }
    auto named = [&m](const std::vector<StoreGame>& libs) {
        std::vector<StoreGame> n;
        for (auto& e : m->games) n.push_back({e.name, e.dir, L"", e.source});
        n.insert(n.end(), libs.begin(), libs.end());
        return n;
    };
    wstring why;
    if (ExistingModelValid(g.ctx, &why)) {
        const std::vector<StoreGame> libs = named(dump || g.noScan ? std::vector<StoreGame>{} : ScanLibraries());  // worker thread: not the Library
        const wstring game = ModelSourceGame(ReadModelSource(g.ctx), &libs);
        m->lines.push_back({0, game.empty() ? L"Model M: Ready" : L"Model M: Ready, converted from " + game + L"'s DLSS file"});
    }
    else if (dump) m->lines.push_back({1, L"Model M: Not converted yet"});
    else {
        const std::vector<StoreGame> libs = g.noScan ? std::vector<StoreGame>{} : ScanLibraries();
        ModelState ms = FindModel(g.ctx, nullptr, dll, libs);
        if (ms.ok()) {
            // converted in memory only: Continue writes it (nothing on this PC changes before the player chooses). The
            // player reads where it comes from; the file's path goes to the log
            LogLine(g.ctx, L"first run: Model M found in " + ms.dll + L" (" + ms.origin + L"); written on Continue");
            m->model = std::make_shared<ModelState>(ms);
            const std::vector<StoreGame> names = named(libs);
            const wstring game = ModelSourceGame(ms.dll, &names);
            m->lines.push_back({0, L"Model M: Found " + (!game.empty() ? game + L"'s DLSS file" : !dll.empty() ? wstring(L"the file you chose")
                                                                                                : wstring(L"a DLSS file on this PC")) +
                                       L". It's converted when you continue."});
        // one short line: the button under the list ("Use another model file...") is the fix; the full reason stays in the log
        } else { LogLine(g.ctx, L"Model M: " + ms.error); m->lines.push_back({2, L"Model M: Needs a recent game's DLSS file (version 310.5\u00a0or\u00a0newer)"}); }
    }
    partial();
    size_t managed = 0, adopted = 0;
    wstring ignored;  // developer / test keys an adoption did not take as settings: "timing_log=1 (The Witcher 3)" (for the log)
    size_t ignoredN = 0;
    for (auto& e : m->games) {
        if (!e.hasRecord) continue;
        ++managed;
        if (e.adoptable && !dump) {
            // adopted on Continue (AdoptGames, elevated where its folder needs it); here only what adopting would keep:
            // the settings its wronggpu.ini has that the launcher doesn't show (read, nothing written)
            m->adopt.push_back(e.dir);
            ++adopted;
            std::vector<wstring> pn;
            if (!IsFile(ProfilePath(e.dir))) {
                string wini, oini;
                ReadAll(Join(e.dir, L"wronggpu.ini"), wini);
                ReadAll(Join(e.dir, L"OptiScaler.ini"), oini);
                ImportProfile(wini, oini, &pn);
            }
            for (auto& n : pn)
                if (n.rfind(kIgnoredPrefix, 0) == 0) {
                    const wstring item = n.substr(wcslen(kIgnoredPrefix));
                    ignored += (ignored.empty() ? L"" : L"; ") + item.substr(0, item.find(L" (")) + L" (" + e.name + L")";
                    ++ignoredN;
                }
        }
    }
    // "14 games, 8 with WrongGPU" ("1 game, 1 with WrongGPU", "14 games, none with WrongGPU yet", "None found yet")
    const size_t found = m->games.size();
    m->lines.push_back({0, L"Games: " + (found == 0 ? wstring(L"None found yet")
                                         : std::to_wstring(found) + (found == 1 ? L" game, " : L" games, ") +
                                               (managed ? std::to_wstring(managed) + L" with WrongGPU" : wstring(L"none with WrongGPU yet")))});
    if (adopted)  // "6 games set up with WrongGPU 0.5 move to the launcher when you continue. Their settings are kept."
        m->lines.push_back({0, L"Earlier installs: " + std::to_wstring(adopted) + (adopted == 1 ? L" game set up with WrongGPU 0.5 moves to the launcher when you continue. Its settings are kept."
                                                                                                : L" games set up with WrongGPU 0.5 move to the launcher when you continue. Their settings are kept.")});
    // developer / test keys an adoption did not take as settings: the log only (nothing for the player to do)
    if (!ignored.empty())
        LogLine(g.ctx, L"first run: " + std::to_wstring(ignoredN) + L" developer/test keys not taken as settings (they stay in wronggpu.ini): " + ignored);
    ReadModelState(m->health);  // the model data as Settings and the toolbar show it, read here (it hashes the file)
    FirstRunMsg* done = m;
    m = nullptr;  // posted here, not by postAnyway
    if (!PostMessageW(hwnd, WM_APP_FIRSTRUN, 0, (LPARAM)done)) delete done;
}

// The first of `exes` running now (its index; -1: none), from one look at the processes: GameRunning's rule for each (the
// program's name, then its full path where Windows tells it; a process that can't be asked counts), without a snapshot per
// game and without reading any game's folder (the window's thread never waits for a game's disk: ReloadEntry).
static int FirstRunningOf(const std::vector<wstring>& exes) {
    if (exes.empty()) return -1;
    HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (s == INVALID_HANDLE_VALUE) return -1;
    int found = -1;
    PROCESSENTRY32W pe{}; pe.dwSize = sizeof pe;
    for (BOOL ok = Process32FirstW(s, &pe); ok && found < 0; ok = Process32NextW(s, &pe))
        for (size_t i = 0; i < exes.size() && found < 0; ++i) {
            if (!IEq(pe.szExeFile, BaseName(exes[i]))) continue;
            HANDLE p = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID);
            if (!p) { found = (int)i; break; }  // cannot tell: assume it is this game (as GameRunning)
            wchar_t path[MAX_PATH * 2]; DWORD n = (DWORD)std::size(path);
            if (!QueryFullProcessImageNameW(p, 0, path, &n) || IEq(path, exes[i])) found = (int)i;
            CloseHandle(p);
        }
    CloseHandle(s);
    return found;
}
// A game that uses the data holds its file while it runs: the data can't be replaced then (said before anything starts,
// before Choose a DLSS file...'s picker too). The games: the Library's with WrongGPU (with its RR files: `rr`), as their
// entries were read (no record read here); the one running now said, true: nothing starts. For the Model M data a game with
// WrongGPU turned off (AMD FSR, wgOff) is not refused on a guess: should it hold the file, the conversion says so in words
// ("couldn't be moved ... a running game uses it", nothing changed).
static bool DataGameRunningSaid(bool rr) {
    std::vector<wstring> exes, names;
    for (auto& e : v.games)
        if (e.hasRecord && !e.exe.empty() && (rr ? e.rrInstalled : !e.wgOff)) { exes.push_back(e.exe); names.push_back(e.name); }
    const int i = FirstRunningOf(exes);
    if (i < 0) return false;
    Inform(L"Close " + names[(size_t)i] + L" first.\n\nIt uses the " + (rr ? L"Ray Reconstruction" : L"Model M") +
               L" data, which can't be replaced while the game runs. The data in use stays.", 1);
    return true;
}
bool RrGameRunningSaid() { return DataGameRunningSaid(true); }
// The same for the Model M data: every game with WrongGPU turned on holds its file while it runs. Settings' buttons wait for a game the
// Library shows running; this catches one started since (asked before the search and the picker). True: said, nothing starts.
bool ModelGameRunningSaid() { return DataGameRunningSaid(false); }

}  // namespace win
}  // namespace wg::app
