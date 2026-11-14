// SPDX-License-Identifier: MIT
// Linux window layer internals shared by win32_window_sdl.cpp (windows, messages, SDL events) and win32_window_edit.cpp (the
// "EDIT" control).
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "win32_window.h"

struct SDL_Window;

namespace wgwin {

// The text field's state (a single line, as the launcher uses it)
struct EditState {
    int anchor = 0, caret = 0;     // the selection: anchor is where it started, caret where it ends (and where the caret shows)
    int marginL = 0, marginR = 0;
    int fontPx = 13;
    int scroll = 0;                // pixels the text is moved left (the caret kept in view)
    std::wstring cue, undoText;
    int undoAnchor = 0, undoCaret = 0;
    bool dragging = false;
    uint64_t lastInputMs = 0;      // the caret shows solid for a moment after an edit or a move
    HFONT font = nullptr;
};

struct Sub { SUBCLASSPROC proc; UINT_PTR id; DWORD_PTR data; };

struct Win {
    uintptr_t id = 0;
    std::wstring cls, text;
    WNDPROC proc = nullptr;        // the class's window procedure; null for a built-in class (EDIT)
    uintptr_t parent = 0;          // 0: top-level (or message-only)
    DWORD style = 0, exStyle = 0;
    bool messageOnly = false, visible = false, builtinEdit = false;
    int x = 0, y = 0, w = 0, h = 0;  // top-level: its place on the screen (pixels); a child: in its parent's client area
    SDL_Window* sdl = nullptr;
    unsigned sdlId = 0;
    float density = 1;             // pixels per SDL window unit (read again when the display scale or pixel size changes)
    RECT dpiRect{};                // the suggested rect of the WM_DPICHANGED message queued for this window
    bool invalid = false;
    uint64_t lastPaintMs = 0;
    bool trackLeave = false, trackNcLeave = false;
    LONG_PTR userdata = 0, ctrlId = 0;
    std::vector<Sub> subs;         // outermost first
    std::unique_ptr<EditState> edit;
    bool maximized = false, minimized = false;
};

Win* Find(HWND h);                 // null when gone
uint64_t NowMs();
// a message posted or sent to a window, through its subclasses to the class's procedure
LRESULT Send(HWND h, UINT m, WPARAM wp, LPARAM lp);
void InvalidateWin(Win* w);
POINT ClientOriginOnScreen(Win* w);   // the client area's top-left corner on the screen (pixels)
HWND TopOf(Win* w);
extern HWND g_focus;               // the window with the keyboard (a text field or the window)
void ShowTextInput(bool on, Win* w);
#ifdef WRONGGPU_TESTING
void TestSetPixelDensity(float d);   // the density every window reports from now on (0: SDL's own); the window units stay
#endif

// the "EDIT" class's procedure (win32_window_edit.cpp)
LRESULT EditClassProc(Win* w, UINT m, WPARAM wp, LPARAM lp);
void EditInit(Win* w);
// the text replaced as a whole with the caret after it (WM_SETTEXT; notify: the undo step and EN_CHANGE of one made by a message)
void EditSetText(Win* w, const std::wstring& text, bool notify);
// measure: the painter's, for caret placement by the mouse
float EditMeasure(const std::wstring& text, int fontPx, int upTo);
std::wstring ClipGet();                  // the clipboard's text (win32_window_sdl.cpp)
void ClipSet(const std::wstring& text);
bool GdiFont(HFONT f, LOGFONTW* out);    // a font made by CreateFontIndirectW (win32_window_misc.cpp)

}  // namespace wgwin
