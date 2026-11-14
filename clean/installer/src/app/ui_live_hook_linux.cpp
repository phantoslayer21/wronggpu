// SPDX-License-Identifier: MIT
// WrongGPU launcher window on Linux, test builds only (WRONGGPU_TESTING): what the window layer's scripted input
// (WRONGGPU_UI_SCRIPT, win32_window_sdl.cpp) can ask of the app. `shot FILE` saves the live frame as it was last drawn (the page
// and its text fields) as a PNG; `dump FILE` writes the window's state and widgets as JSON, so a test can check that a click, a
// scroll and typing changed the page. A release build has neither: it never reads a test variable.
#ifdef WRONGGPU_TESTING
#include "app/launcher_window.h"

#include "wg_core/json_writer.h"

namespace wg::app {
using namespace wg::app::lui;
namespace win {
void DumpHits(Json& j);  // ui_dump.cpp
extern Gfx gfx;          // launcher_ui.cpp

namespace {
bool LiveHook(const std::string& line) {
    const size_t sp = line.find(' ');
    const std::string cmd = line.substr(0, sp);
    const wstring file = sp == std::string::npos ? wstring() : FromUtf8(line.substr(sp + 1));
    if (file.empty()) return false;
    if (cmd == "until") {  // until games N: the scan has listed N games
        int n = 0;
        return swscanf(file.c_str(), L"games %d", &n) == 1 && !v.scanning && (int)v.games.size() >= n;
    }
    if (cmd == "clickname") {  // the middle of the first shown widget whose name starts with this
        for (const auto& h : v.hits)
            if (h.shown && h.name.rfind(file, 0) == 0) { wgwin::ScriptClick(h.r.cx(), h.r.cy()); return true; }
        fprintf(stderr, "wronggpu: no widget named %s\n", line.c_str());
        return false;
    }
    if (cmd == "shot") return gfx.SavePng(file);
    Json j;
    j.Obj().N("page", (long long)v.page).S("page_name", v.page == P_LIBRARY ? L"library" : v.page == P_GAME ? L"game" : L"settings")
        .N("scroll", (long long)std::lround(v.scroll[v.page])).N("content_h", (long long)std::lround(v.contentH[v.page]))
        .S("search", v.search).S("url", v.urlText).N("focus", v.focus).B("dialog_open", v.dialog.open).B("menu_open", v.menu.open)
        .B("first_run_open", v.first.open).N("games", (long long)v.games.size()).N("w", (long long)std::lround(v.W)).N("h", (long long)std::lround(v.H));
    j.Key("controls");
    DumpHits(j);
    j.End();
    const string s = j.str() + "\n";
    try { WriteFileAtomic(file, s.data(), s.size()); } catch (const SetupError&) { return false; }
    return true;
}
[[maybe_unused]] const bool kRegistered = (wgwin::SetScriptHook(LiveHook), true);
}  // namespace
}  // namespace win
}  // namespace wg::app
#endif
