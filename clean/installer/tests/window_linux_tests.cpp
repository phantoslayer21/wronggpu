// SPDX-License-Identifier: MIT
// Linux: the window layer's message handling (src/platform/posix/win32_window_*.cpp) against SDL's offscreen driver, with SDL events
// pushed by the test: a display scale change, a batch of key events, text typed into an edit, WM_QUIT against due paints and timers,
// posts from another thread, SetCapture and PeekMessageW's filters. No window opens on a display.
#include <SDL3/SDL.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <thread>

#include "win32_window_internal.h"

using namespace wgwin;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)
#define CHECK_EQ(a, b) do { const long long x_ = (long long)(a), y_ = (long long)(b); if (x_ != y_) { std::fprintf(stderr, "FAIL line %d: %s == %s (%lld vs %lld)\n", __LINE__, #a, #b, x_, y_); ++fails; } } while (0)

constexpr UINT kUser = 0x0400;

struct Seen {
    RECT dpiRect{};
    int dpiCount = 0;
    UINT dpi = 0;
    LPARAM lastDown = 0;
    int downCount = 0;
    LPARAM lastSize = 0;
    int captureLost = 0;
    LPARAM captureTo = 0;
    int paints = 0;
};
static std::map<HWND, Seen> seen;

static LRESULT CALLBACK TestProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    Seen& s = seen[h];
    switch (m) {
        case WM_DPICHANGED: {  // what the launcher's window procedure does
            const RECT* r = (const RECT*)lp;
            s.dpiRect = *r; ++s.dpiCount; s.dpi = HIWORD(wp);
            SetWindowPos(h, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
            return 0;
        }
        case WM_LBUTTONDOWN: s.lastDown = lp; ++s.downCount; return 0;
        case WM_SIZE: s.lastSize = lp; return 0;
        case WM_CAPTURECHANGED: ++s.captureLost; s.captureTo = lp; return 0;
        case WM_PAINT: ++s.paints; return DefWindowProcW(h, m, wp, lp);
        default: return DefWindowProcW(h, m, wp, lp);
    }
}

static HWND Top(int w, int h) {
    HWND hw = CreateWindowExW(0, L"WgTest", L"t", WS_THICKFRAME | WS_VISIBLE, 0, 0, w, h, nullptr, nullptr, nullptr, nullptr);
    ShowWindow(hw, SW_SHOW);
    return hw;
}
static HWND Edit(HWND parent, const wchar_t* text, int x, int y, int w, int h) {
    return CreateWindowExW(0, L"EDIT", text, WS_CHILD | WS_VISIBLE, x, y, w, h, parent, (HMENU)(uintptr_t)7, nullptr, nullptr);
}

// everything that is ready, handled as the app's loop does
static void Pump() {
    MSG m;
    int guard = 0;
    while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE) && ++guard < 10000) {
        if (m.message == WM_QUIT) return;
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
}

static unsigned SdlId(HWND h) { return Find(h)->sdlId; }
static void PushWin(SDL_EventType t, HWND h) {
    SDL_Event e;
    SDL_zero(e);
    e.type = t;
    e.window.windowID = SdlId(h);
    SDL_PushEvent(&e);
}
static void PushKey(HWND h, SDL_Keycode k, SDL_Keymod mod, bool down) {
    SDL_Event e;
    SDL_zero(e);
    e.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
    e.key.windowID = SdlId(h);
    e.key.key = k; e.key.mod = mod; e.key.down = down;
    SDL_PushEvent(&e);
}
static void PushMouse(HWND h, SDL_EventType t, float x, float y) {
    SDL_Event e;
    SDL_zero(e);
    e.type = t;
    if (t == SDL_EVENT_MOUSE_MOTION) { e.motion.windowID = SdlId(h); e.motion.x = x; e.motion.y = y; }
    else { e.button.windowID = SdlId(h); e.button.x = x; e.button.y = y; e.button.button = SDL_BUTTON_LEFT; e.button.clicks = 1; e.button.down = t == SDL_EVENT_MOUSE_BUTTON_DOWN; }
    SDL_PushEvent(&e);
}
static void PushText(HWND h, const char* text) {
    static std::string keep;
    keep = text;
    SDL_Event e;
    SDL_zero(e);
    e.type = SDL_EVENT_TEXT_INPUT;
    e.text.windowID = SdlId(h);
    e.text.text = keep.c_str();
    SDL_PushEvent(&e);
}
static std::wstring Text(HWND h) {
    wchar_t b[256];
    GetWindowTextW(h, b, 256);
    return b;
}
static void Sel(HWND h, int& a, int& b) {
    DWORD s = 0, e = 0;
    SendMessageW(h, EM_GETSEL, (WPARAM)&s, (LPARAM)&e);
    a = (int)s; b = (int)e;
}

// 1. a display scale change: the density is read again before the suggested rect and the mouse use it
static void TestDpi() {
    HWND a = Top(400, 300), b = Top(300, 200);
    HWND field = Edit(a, L"", 180, 80, 200, 40);
    Pump();
    seen.clear();
    TestSetPixelDensity(2.f);
    PushWin(SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED, a);  // both queued before either is handled: each has its own rect
    PushWin(SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED, b);
    Pump();
    CHECK_EQ(seen[a].dpiCount, 1);
    CHECK_EQ(seen[b].dpiCount, 1);
    CHECK_EQ(seen[a].dpi, 192);
    CHECK_EQ(seen[a].dpiRect.right - seen[a].dpiRect.left, 800);  // the SDL window kept its units: 400 x 2 pixels
    CHECK_EQ(seen[a].dpiRect.bottom - seen[a].dpiRect.top, 600);
    CHECK_EQ(seen[b].dpiRect.right - seen[b].dpiRect.left, 600);
    CHECK_EQ(seen[b].dpiRect.bottom - seen[b].dpiRect.top, 400);
    RECT rc{};
    GetClientRect(a, &rc);
    CHECK_EQ(rc.right, 800);   // SetWindowPos with the suggested rect: not scaled twice (1600)
    CHECK_EQ(rc.bottom, 600);
    GetClientRect(b, &rc);
    CHECK_EQ(rc.right, 600);
    CHECK_EQ(rc.bottom, 400);
    // a click at SDL unit (100, 50) is pixel (200, 100): inside the field at x 180..380, y 80..120
    PushMouse(a, SDL_EVENT_MOUSE_MOTION, 100, 50);
    PushMouse(a, SDL_EVENT_MOUSE_BUTTON_DOWN, 100, 50);
    PushMouse(a, SDL_EVENT_MOUSE_BUTTON_UP, 100, 50);
    Pump();
    CHECK(GetFocus() == field);
    // the pixel size alone changing (the compositor scaled the window to another density) is read the same way, with no WM_DPICHANGED
    TestSetPixelDensity(3.f);
    seen[a].dpiCount = 0;
    PushWin(SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED, a);
    Pump();
    CHECK_EQ(LOWORD(seen[a].lastSize), 1200);
    seen[a].downCount = 0;
    PushMouse(a, SDL_EVENT_MOUSE_MOTION, 100, 50);
    PushMouse(a, SDL_EVENT_MOUSE_BUTTON_DOWN, 100, 50);
    PushMouse(a, SDL_EVENT_MOUSE_BUTTON_UP, 100, 50);
    Pump();
    CHECK_EQ(seen[a].downCount, 1);
    CHECK_EQ(GET_X_LPARAM(seen[a].lastDown), 300);   // pixel = unit x 3 (the field has the focus: a click on it would not reach the window)
    CHECK_EQ(GET_Y_LPARAM(seen[a].lastDown), 150);
    TestSetPixelDensity(0.f);
    DestroyWindow(a);
    DestroyWindow(b);
}

// 2. Ctrl down, A down, A up, Ctrl up in one batch: the A key is handled with Ctrl down
static void TestKeyBatch() {
    HWND w = Top(400, 300);
    HWND e = Edit(w, L"abc", 10, 10, 200, 30);
    Pump();
    SetFocus(e);
    SendMessageW(e, EM_SETSEL, 1, 1);
    PushKey(w, SDLK_LCTRL, SDL_KMOD_LCTRL, true);
    PushKey(w, SDLK_A, SDL_KMOD_LCTRL, true);
    PushKey(w, SDLK_A, SDL_KMOD_LCTRL, false);
    PushKey(w, SDLK_LCTRL, SDL_KMOD_NONE, false);
    Pump();
    int a = -1, b = -1;
    Sel(e, a, b);
    CHECK_EQ(a, 0);
    CHECK_EQ(b, 3);
    CHECK(GetKeyState(VK_CONTROL) >= 0);  // the last message handled was the Ctrl release
    DestroyWindow(w);
}

// 3. the caret after the text is set is at its start, as in a Win32 edit (typing goes in front)
static void TestCaret() {
    HWND w = Top(400, 300);
    HWND e = Edit(w, L"0.5", 10, 10, 200, 30);
    Pump();
    SetFocus(e);
    PushText(w, "1");
    Pump();
    CHECK(Text(e) == L"10.5");
    SetWindowTextW(e, L"xyz");
    int a = -1, b = -1;
    Sel(e, a, b);
    CHECK_EQ(a, 0);
    CHECK_EQ(b, 0);
    PushText(w, "!");
    Pump();
    CHECK(Text(e) == L"!xyz");
    DestroyWindow(w);
}

// 4a. WM_QUIT comes before a paint or a timer that is due
static void TestQuitFirst() {
    HWND w = Top(200, 100);
    Pump();
    InvalidateRect(w, nullptr, FALSE);
    SetTimer(w, 9, 1, nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    PostQuitMessage(7);
    MSG m{};
    int others = 0;
    bool quit = false;
    for (int i = 0; i < 50 && !quit; ++i) {
        if (!PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) break;
        if (m.message == WM_QUIT) { quit = true; CHECK_EQ(m.wParam, 7); }
        else if (m.message == WM_PAINT || m.message == WM_TIMER) ++others;
    }
    CHECK(quit);
    CHECK_EQ(others, 0);
    KillTimer(w, 9);
    DestroyWindow(w);
}

// 4b. a post from another thread carries no pointer position (the UI thread's is not read across threads)
static void TestCrossThreadPost() {
    HWND w = Top(400, 300);
    Pump();
    PushMouse(w, SDL_EVENT_MOUSE_MOTION, 50, 60);
    Pump();
    POINT p{};
    GetCursorPos(&p);
    CHECK(p.x != 0 || p.y != 0);
    std::thread([&] { PostMessageW(w, kUser + 1, 0, 0); }).join();
    PostMessageW(w, kUser + 2, 0, 0);
    MSG m;
    bool a = false, b = false;
    while (PeekMessageW(&m, nullptr, 0, 0, PM_REMOVE)) {
        if (m.message == kUser + 1) { a = true; CHECK(m.pt.x == 0 && m.pt.y == 0); }
        if (m.message == kUser + 2) { b = true; CHECK(m.pt.x == p.x && m.pt.y == p.y); }
    }
    CHECK(a && b);
    DestroyWindow(w);
}

// 4d. taking the capture from another window tells the first one
static void TestCapture() {
    HWND a = Top(200, 100), b = Top(200, 100);
    Pump();
    seen.clear();
    SetCapture(a);
    SetCapture(b);
    CHECK_EQ(seen[a].captureLost, 1);
    CHECK((HWND)seen[a].captureTo == b);
    SetCapture(b);  // the same window again: nothing lost
    CHECK_EQ(seen[b].captureLost, 0);
    ReleaseCapture();
    DestroyWindow(a);
    DestroyWindow(b);
}

// 4e. PeekMessageW's window, range and PM_NOREMOVE for the paints and timers it makes up
static void TestPeekFilters() {
    HWND a = Top(200, 100), b = Top(200, 100);
    Pump();
    std::this_thread::sleep_for(std::chrono::milliseconds(30));  // past the frame gap
    Pump();
    InvalidateRect(a, nullptr, FALSE);
    MSG m{};
    CHECK(!PeekMessageW(&m, b, WM_PAINT, WM_PAINT, PM_NOREMOVE));          // b has nothing to paint
    CHECK(!PeekMessageW(&m, nullptr, kUser, kUser + 10, PM_NOREMOVE));     // a paint is not in this range
    CHECK(PeekMessageW(&m, a, WM_PAINT, WM_PAINT, PM_NOREMOVE) && m.hwnd == a && m.message == WM_PAINT);
    CHECK(PeekMessageW(&m, nullptr, WM_PAINT, WM_PAINT, PM_NOREMOVE));     // and still there: not taken
    CHECK(PeekMessageW(&m, a, 0, 0, PM_REMOVE) && m.message == WM_PAINT);
    DispatchMessageW(&m);
    CHECK(!PeekMessageW(&m, a, WM_PAINT, WM_PAINT, PM_NOREMOVE));          // painted: no longer invalid
    SetTimer(a, 9, 20, nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    CHECK(!PeekMessageW(&m, b, WM_TIMER, WM_TIMER, PM_NOREMOVE));
    CHECK(PeekMessageW(&m, a, WM_TIMER, WM_TIMER, PM_NOREMOVE) && m.wParam == 9);
    CHECK(PeekMessageW(&m, a, WM_TIMER, WM_TIMER, PM_NOREMOVE));           // left queued: it did not move to its next period
    CHECK(PeekMessageW(&m, a, WM_TIMER, WM_TIMER, PM_REMOVE));
    CHECK(!PeekMessageW(&m, a, WM_TIMER, WM_TIMER, PM_NOREMOVE));          // taken: due again in 20 ms
    KillTimer(a, 9);
    DestroyWindow(a);
    DestroyWindow(b);
}

int main() {
    setenv("SDL_VIDEO_DRIVER", "offscreen", 1);  // never a window on the player's desktop
    unsetenv("XDG_RUNTIME_DIR");
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof wc; wc.lpfnWndProc = TestProc; wc.lpszClassName = L"WgTest";
    RegisterClassExW(&wc);
    TestDpi();
    TestKeyBatch();
    TestCaret();
    TestCrossThreadPost();
    TestCapture();
    TestPeekFilters();
    TestQuitFirst();
    if (fails) { std::fprintf(stderr, "window-linux-tests: %d failure(s)\n", fails); return 1; }
    std::printf("window-linux-tests: ok\n");
    return 0;
}
