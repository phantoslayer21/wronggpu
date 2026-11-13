// SPDX-License-Identifier: MIT
// WrongGPU launcher window: Ray Reconstruction's kernels prepared by the launcher (release 0.6.0, owner 2026-11-02: "It HAS
// to pre compile inside the launcher"; wg_core/rrprep.h). After an install, update or repair of a game with Ray
// Reconstruction, at Play when they aren't prepared (a driver or WrongGPU update since), and with the game page's Prepare.
// The game's lockup shows "Preparing Ray Reconstruction (n of m)"; Esc cancels (what was compiled stays with the driver:
// the next run goes on from there).
//
// Release 0.6.0 polish (owner 2026-11-02: an AMD driver update clears the driver's shader cache): at the window's start
// (after its first scan) the games prepared before whose graphics driver or WrongGPU runtime changed since
// (RrStaleAfterUpdate) are prepared again in the BACKGROUND, one after the other, every game's default mode first, then
// the other passes their modes need. The toolbar's status line says "Preparing Ray Reconstruction after the driver update
// (n of m)" meanwhile, and their game page says not prepared until each is done. Anything the window starts itself (an
// operation, a read, Play's or the page's own prepare) stops the background run first (the helper holds the game's runtime
// DLL; what was compiled stays with the driver) and it goes on after that. Games that prepare their kernels at their own
// first start (gamerules.h rrKernelsInGame, RE Requiem) never run the helper. Written for this project.
#include "app/launcher_app.h"
#include "app/launcher_window.h"

#include <algorithm>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace wg::app {
using namespace wg::app::lui;
namespace win {

// as launcher_ui.cpp has them (the window's step message, the background run's message)
const UINT WM_APP_STEP = WM_APP + 22;
const UINT WM_APP_RRPREP = WM_APP + 48;
const UINT WM_APP_RRBG = WM_APP + 49;
struct StepMsg { int done, total; wstring text; };

void NoteLater(const wstring& text, int icon, const wstring& details);
std::atomic<bool> rrPrepCancel{false};
bool rrPrepRunning = false;
std::map<wstring, std::pair<bool, wstring>> rrReadyCache;  // cleared by every prepare
std::vector<std::pair<wstring, wstring>> rrPrepQueue;      // dir, exe

// ---- the background run (window's start) ------------------------------------------------------------------------------
struct RrBgJob { wstring dir, exe, runtime; int passes = 1; bool driver = true; };
struct RrBg {
    std::thread worker;
    std::atomic<bool> cancel{false}, running{false};
    std::mutex mu;               // guards the fields below (the worker writes them, the window reads them)
    std::vector<RrBgJob> jobs;   // still to do (the one running included, until it is done)
    wstring curExe;              // the game being prepared now
    unsigned done = 0, total = 0;
    bool driverWords = true;     // "after the driver update" (else "after the WrongGPU update")
    bool wanted = false;         // stopped for something the window started: goes on after it
    bool checked = false;        // the start's check ran (once per session; again after a stop)
};
RrBg rrBg;

bool RrGameInstalled(const wstring& dir) { return !dir.empty() && IsFile(Join(dir, kRrIni)); }
wstring RrRuntimeFor(const wstring& dir, const wstring& exe) {
    const wstring a = Join(Parent(exe), kRrRuntime);
    return IsFile(a) ? a : Join(dir, kRrRuntime);
}
// the background run has this game still to do (or does it now)
bool RrBackgroundFor(const wstring& exe) {
    if (!rrBg.running.load()) return false;
    std::lock_guard<std::mutex> lock(rrBg.mu);
    for (auto& j : rrBg.jobs) if (IEq(j.exe, exe)) return true;
    return false;
}
bool RrReadyForGame(const wstring& dir, const wstring& exe, wstring* why) {
    if (v.fixture) {  // ui-render: the fixture says which game is prepared (r5 #8), never this PC's prepared.txt
        if (why) *why = L"not prepared yet";
        return !v.fixtureRrPrepared.empty() && IEq(v.fixtureRrPrepared, exe);
    }
    if (RrBackgroundFor(exe)) {
        std::lock_guard<std::mutex> lock(rrBg.mu);
        if (why) *why = rrBg.driverWords ? L"being prepared again after the driver update" : L"being prepared again after the WrongGPU update";
        return false;
    }
    const wstring k = Lower(exe) + L"|" + Lower(dir);
    auto it = rrReadyCache.find(k);
    if (it == rrReadyCache.end()) {
        wstring w;
        const bool ok = RrPrepared(g.ctx, gl.paths.dataDir, exe, RrPassesFor(ReadRrSettings(dir).mode), &w);
        it = rrReadyCache.emplace(k, std::make_pair(ok, w)).first;
    }
    if (why) *why = it->second.second;
    return it->second.first;
}

// The toolbar's status line part while the background run goes (empty: none).
wstring RrBackgroundStatus() {
    if (!rrBg.running.load()) return L"";
    std::lock_guard<std::mutex> lock(rrBg.mu);
    wstring s = rrBg.driverWords ? L"Preparing Ray Reconstruction after the driver update" : L"Preparing Ray Reconstruction after the WrongGPU update";
    if (rrBg.total) s += L" (" + std::to_wstring(std::min(rrBg.done, rrBg.total)) + L" of " + std::to_wstring(rrBg.total) + L")";
    return s;
}
// What the game page says for a game the background run has (empty: not one of them).
wstring RrBackgroundWords(const wstring& exe) {
    if (!RrBackgroundFor(exe)) return L"";
    std::lock_guard<std::mutex> lock(rrBg.mu);
    const wstring after = rrBg.driverWords ? L"after the driver update" : L"after the WrongGPU update";
    if (!IEq(rrBg.curExe, exe)) return L"Waiting to be prepared " + after;
    return L"Preparing " + after + (rrBg.total ? L" (" + std::to_wstring(std::min(rrBg.done, rrBg.total)) + L" of " + std::to_wstring(rrBg.total) + L")" : L"");
}

// Stops the background run (the window starts something of its own); it goes on after that (RunRrPrepQueue).
void StopRrBackground() {
    if (!rrBg.worker.joinable()) return;
    {
        std::lock_guard<std::mutex> lock(rrBg.mu);
        if (rrBg.running.load() && !rrBg.jobs.empty()) rrBg.wanted = true;
    }
    rrBg.cancel = true;
    rrBg.worker.join();  // PrepareRr looks at its progress every 100 ms and ends the helper at once
    rrBg.cancel = false;
    rrBg.running = false;
    rrReadyCache.clear();
}

void StartRrRecheck() {
    if (!v.live || v.fixture || rrBg.running.load()) return;
    if (rrBg.checked && !rrBg.wanted) return;
    rrBg.checked = true;
    rrBg.wanted = false;
    if (rrBg.worker.joinable()) rrBg.worker.join();
    // the installed RR games (their files, the driver's version and the prepared records are read on the worker, never here)
    std::vector<RrBgJob> cand;
    for (auto& e : v.games)
        // rc5: not a game whose RR waits for its experimental opt-in, nor one whose menu has no RR option (gamerules.h)
        if (e.hasRecord && e.rrInstalled && e.rrUses && !e.rrHeldOff && e.exeFound && !RrKernelsInGame(e.exe)) cand.push_back({e.dir, e.exe, L"", 1, true});
    if (cand.empty()) return;
    // a game running: not now (the helper would take GPU time from it); the next scan or operation's end looks again
    for (auto& e : v.games) if (e.status == GameStatus::Running) { rrBg.checked = false; return; }
    const wstring data = gl.paths.dataDir;
    HWND hwnd = v.hwnd;
    rrBg.running = true;
    rrBg.worker = std::thread([cand, data, hwnd]() mutable {
        // which of them were prepared before and lost it to a driver / runtime change (the default mode's kernels)
        std::vector<RrBgJob> jobs;
        for (auto& c : cand) {
            if (rrBg.cancel.load()) break;
            if (!RrGameInstalled(c.dir)) continue;
            c.runtime = RrRuntimeFor(c.dir, c.exe);
            c.passes = RrPassesFor(ReadRrSettings(c.dir).mode);
            wstring why;
            const RrPrepState s = RrPrepStateOf(g.ctx, data, c.exe, 1, &why);
            if (!RrStaleAfterUpdate(s)) continue;
            c.driver = s == RrPrepState::DriverChanged;
            LogLine(g.ctx, L"Ray Reconstruction: " + BaseName(c.exe) + L" is prepared again in the background (" + why + L")");
            jobs.push_back(c);
        }
        {
            std::lock_guard<std::mutex> lock(rrBg.mu);
            rrBg.jobs = jobs;
            rrBg.done = rrBg.total = 0;
            rrBg.driverWords = jobs.empty() || std::any_of(jobs.begin(), jobs.end(), [](const RrBgJob& j) { return j.driver; });
        }
        PostMessageW(hwnd, WM_APP_RRBG, 0, 0);
        // every game's default mode first, then the other passes the games' modes need
        for (int round = 0; round < 2 && !rrBg.cancel.load(); ++round) {
            for (auto& j : jobs) {
                if (rrBg.cancel.load()) break;
                const int passes = round == 0 ? 1 : j.passes;
                if (round == 1 && passes <= 1) continue;
                {
                    std::lock_guard<std::mutex> lock(rrBg.mu);
                    rrBg.curExe = j.exe;
                    rrBg.done = 0; rrBg.total = 0;
                }
                PostMessageW(hwnd, WM_APP_RRBG, 0, 0);
                ULONGLONG lastPost = 0;
                wstring why;
                const int rc = PrepareRr(g.ctx, data, j.exe, j.runtime, passes, [hwnd, &lastPost](const RrPrepProgress& p) {
                    {
                        std::lock_guard<std::mutex> lock(rrBg.mu);
                        rrBg.done = p.done; rrBg.total = p.total;
                    }
                    const ULONGLONG now = GetTickCount64();
                    if (now - lastPost >= 250) { lastPost = now; PostMessageW(hwnd, WM_APP_RRBG, 0, 0); }
                    return !rrBg.cancel.load();
                }, &why);
                if (rc == 3) LogLine(g.ctx, L"Ray Reconstruction: preparing " + BaseName(j.exe) + L" again failed in the background (" + why + L"); Play prepares it");
                // done with this game unless its other passes follow in the second round
                if (rc != 1 && (round == 1 || j.passes <= 1)) {
                    std::lock_guard<std::mutex> lock(rrBg.mu);
                    rrBg.jobs.erase(std::remove_if(rrBg.jobs.begin(), rrBg.jobs.end(), [&j](const RrBgJob& x) { return IEq(x.exe, j.exe); }), rrBg.jobs.end());
                }
                PostMessageW(hwnd, WM_APP_RRBG, 1, 0);  // a game finished: its page reads its state again
            }
        }
        {
            std::lock_guard<std::mutex> lock(rrBg.mu);
            if (!rrBg.cancel.load()) rrBg.jobs.clear();
            rrBg.curExe.clear();
        }
        rrBg.running = false;
        PostMessageW(hwnd, WM_APP_RRBG, 2, 0);
    });
}

// WM_APP_RRBG: the background run's progress (0), a game done (1), the run ended (2)
void RrBackgroundMessage(WPARAM wp) {
    if (wp >= 1) rrReadyCache.clear();
    if (wp == 2 && rrBg.worker.joinable() && !rrBg.running.load()) rrBg.worker.join();
    Invalidate();
}

// False only when the window is closing.
bool PrepareRrNow(const wstring& dir, const wstring& exe) {
    if (!RrGameInstalled(dir)) return true;
    if (RrKernelsInGame(exe)) return true;  // the game prepares them at its own first start (rrprep.h)
    StopRrBackground();  // one helper at a time (it holds the game's runtime DLL); the background run goes on after this
    const int passes = RrPassesFor(ReadRrSettings(dir).mode);
    auto res = std::make_shared<std::pair<int, wstring>>(3, L"");
    const wstring data = gl.paths.dataDir, rt = RrRuntimeFor(dir, exe);
    HWND hwnd = v.hwnd;
    const bool live = v.live && v.hwnd;
    rrPrepCancel = false;
    rrPrepRunning = true;
    const bool ok = WaitForRead(dir, L"Preparing Ray Reconstruction…", T_NONE, [res, data, rt, exe, passes, hwnd, live] {
        int last = -1;
        res->first = PrepareRr(g.ctx, data, exe, rt, passes, [&last, hwnd, live](const RrPrepProgress& p) {
            if (live && p.total && (int)p.done != last) {
                last = (int)p.done;
                auto* m = new StepMsg{(int)p.done, (int)p.total, L"Preparing Ray Reconstruction kernels"};
                if (!PostMessageW(hwnd, WM_APP_STEP, 0, (LPARAM)m)) delete m;
            }
            return !rrPrepCancel.load();
        }, &res->second);
    });
    rrPrepRunning = false;
    rrReadyCache.clear();
    if (!ok) return false;
    if (res->first == 3)
        NoteLater(L"Ray Reconstruction couldn't be prepared\n\nThe first start in the game prepares it instead. That can take a few minutes; until then "
                  L"OptiScaler's overlay says \"WrongGPU RR (preparing...)\".", 1, res->second);
    Invalidate();
    return true;
}
// the games waiting for it, once nothing else runs (an operation's end posts WM_APP_RRPREP); then the background run goes on
void RunRrPrepQueue() {
    while (!rrPrepQueue.empty() && !v.busy && v.queue.empty()) {
        const auto job = rrPrepQueue.front();
        rrPrepQueue.erase(rrPrepQueue.begin());
        if (!RrKernelsInGame(job.second) && !RrReadyForGame(job.first, job.second) && !PrepareRrNow(job.first, job.second)) return;
    }
    if (rrPrepQueue.empty() && !v.busy && v.queue.empty()) StartRrRecheck();
}

bool RrPrepRunningNow() { return rrPrepRunning; }
void CancelRrPrep() { rrPrepCancel = true; }
bool RrPrepCancelled() { return rrPrepCancel.load(); }
void QueueRrPrep(const wstring& dir, const wstring& exe) {
    rrReadyCache.clear();
    rrPrepQueue.push_back({dir, exe});
}
// Play: false when the game shouldn't start now (the window is closing, or the player cancelled the prepare)
bool RrReadyBeforePlay(const wstring& dir, const wstring& exe) {
    // rc6: no RR prepare where WrongGPU's RR isn't offered (the updated Hogwarts Legacy: the entry's rrNotOffered)
    for (auto& e : v.games) if (IEq(e.dir, dir) && IEq(e.exe, exe) && e.rrNotOffered) return true;
    if (IEq(v.cur.dir, dir) && IEq(v.cur.exe, exe) && v.cur.rrNotOffered) return true;
    if (!RrGameInstalled(dir) || RrKernelsInGame(exe) || RrReadyForGame(dir, exe)) return true;
    return PrepareRrNow(dir, exe) && !RrPrepCancelled();
}

}  // namespace win
}  // namespace wg::app
