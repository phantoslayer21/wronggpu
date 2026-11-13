// SPDX-License-Identifier: MIT
// WrongGPU launcher window: the running games (launcher_ui.cpp has Play, which starts a game on a worker): the running-game
// watch, the games Play started (a store may start one again in a moment), a game shown running at Play's press and as it
// was again when Windows didn't start it, with a note that says why. Split from launcher_ui.cpp (source size). Written for
// this project.
#include "app/launcher_window.h"

namespace wg::app {
using namespace wg::app::lui;

namespace win {

void NoteLater(const wstring& text, int icon, const wstring& details);  // launcher_ui.cpp: a note once no dialog is open

// The running-game watch (timer 4, every 3 s) while any game in the Library runs: started by Play, and by a scan or an
// operation that finds one running (a game started from Steam), so its page unlocks once it closes
void WatchRunning() {
    if (!v.live || !v.hwnd) return;
    for (auto& e : v.games) if (e.status == GameStatus::Running) { SetTimer(v.hwnd, 4, 3000, nullptr); return; }
}
// The games Play started and when, by exe in lower case. A Steam game started by its exe often closes at once and Steam starts
// it again seconds later: for kLaunchGraceMs neither the running-game watch nor a scan shows it closed (no second start).
std::map<wstring, ULONGLONG> launchedAt;
constexpr ULONGLONG kLaunchGraceMs = 15000;
bool Launching(const wstring& exe) {
    const auto it = launchedAt.find(Lower(exe));
    return it != launchedAt.end() && GetTickCount64() - it->second < kLaunchGraceMs;
}
// rc6: the games Play started, by exe in lower case, until the watch saw them close: when, and whether seen running yet. A
// store's own launcher may start the game only when the player says so there (Cyberpunk 2077 through Steam: REDlauncher, 28 s
// after Play): until it is seen running it is starting (WatchPlayed), never taken as closed before it ran, which read its
// entry before the session and stopped watching, so Last run never showed that session
std::map<wstring, PlayedGame> played;
bool AwaitingStart(const wstring& exe) {
    const auto it = played.find(Lower(exe));
    return it != played.end() && !it->second.seen && WatchPlayed(GetTickCount64() - it->second.at, false, false) == PlayWatch::Starting;
}
// Entries read during that gap (a scan's, a second read's) shown running again, as Play showed them
void KeepLaunching() {
    for (auto it = launchedAt.begin(); it != launchedAt.end();) it = GetTickCount64() - it->second >= kLaunchGraceMs ? launchedAt.erase(it) : std::next(it);
    if (launchedAt.empty() && played.empty()) return;
    auto starting = [](const wstring& exe) { return Launching(exe) || AwaitingStart(exe); };
    for (auto& x : v.games)
        if (x.status != GameStatus::Running && starting(x.exe)) { x.status = GameStatus::Running; x.statusText = L"Running (actions locked)"; }
    if (v.page == P_GAME && v.cur.status != GameStatus::Running && starting(v.cur.exe)) { v.cur.status = GameStatus::Running; v.cur.statusText = L"Running (actions locked)"; }
    WatchRunning();
}
// The game shown running (its lockup, its page): at Play's press, before Windows is asked to start it
void ShowRunning(const wstring& exe) {
    for (auto& x : v.games) if (IEq(x.exe, exe)) { x.status = GameStatus::Running; x.statusText = L"Running (actions locked)"; }
    if (v.page == P_GAME && IEq(v.cur.exe, exe)) { v.cur.status = GameStatus::Running; v.cur.statusText = L"Running (actions locked)"; }
}
// Play's answer from Windows (Play's worker, or at once without a window): `ok` started; `running` it was running
// already; `gone` its program is missing; `why` the core's words. `was`: the game as it was before Play showed it running.
void PlayEnded(const GameEntry& was, bool ok, bool running, bool gone, const wstring& why) {
    if (!ok && !running) {  // not started: the game as it was (its lockup and page unlock), and launchedAt forgets it
        launchedAt.erase(Lower(was.exe));
        played.erase(Lower(was.exe));
        auto back = [&](GameEntry& x) {
            if (IEq(x.exe, was.exe) && x.status == GameStatus::Running) { x.status = was.status; x.statusText = was.statusText; }
        };
        for (auto& x : v.games) back(x);
        if (v.page == P_GAME) back(v.cur);
        Invalidate();
    }
    if (!ok && why.rfind(L"cancelled", 0) == 0) return;  // the player said no to Windows' administrator prompt: their choice, no note
    if (!ok && !running) {  // what happened in words, the core's text behind Show details (a note: once no dialog is open)
        const wstring store = was.source == L"Steam" ? L"Steam" : was.source == L"Epic" ? L"the Epic Games Launcher" : was.source == L"GOG" ? L"GOG Galaxy" : L"";
        NoteLater(was.name + L" couldn't be started.\n\n" +
                      (gone ? L"The game's program is missing: the game may have moved or been uninstalled. Rescan the Library to find it again."
                            : L"Windows couldn't start it. " + (store.empty() ? wstring(L"Try starting it from its own shortcut.") : L"Try starting it from " + store + L".")),
                  kIconError, why);
        return;
    }
    if (!ok) ShowToast(was.name + L" is running already.", 0);  // started some other way: shown as running
    Invalidate();
}

}  // namespace win
}  // namespace wg::app
