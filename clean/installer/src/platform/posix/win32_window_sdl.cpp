// SPDX-License-Identifier: MIT
// Linux: the Win32 window layer on SDL3 (win32_window.h). The launcher's WndProc, its timers, its posted messages from worker
// threads and its nested message loops (dialogs) all run on it unchanged: SDL events become the Win32 messages the app answers,
// posted messages and timers share one queue, WM_PAINT comes when the queue is empty, and the custom caption works through
// WM_NCHITTEST (SDL_SetWindowHitTest moves and sizes a borderless window; the caption buttons arrive as WM_NC* messages).
// Without SDL's video (ui-dump and ui-render use a message-only window) nothing here starts SDL.
#include "win32_window_internal.h"

#include <SDL3/SDL.h>
#include <fcntl.h>
#include <sys/file.h>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <thread>
#include <unordered_map>

#include "win32_posix_internal.h"

namespace wgwin {

HWND g_focus = nullptr;
uint64_t NowMs() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

namespace {

std::mutex g_mu;
std::unordered_map<uintptr_t, std::unique_ptr<Win>> g_wins;
uintptr_t g_nextId = 1;
std::map<std::wstring, WNDCLASSEXW> g_classes;  // by lower-case name
// The keys and buttons down as of an input message: GetKeyState reports these for the message being handled (Win32), not what SDL's
// events later in the same batch made of the keyboard.
struct KeySnap { std::bitset<256> down; uint8_t btn = 0; };
struct Queued { MSG msg; bool hasSnap; KeySnap snap; };
std::deque<Queued> g_q;
std::condition_variable g_cv;
bool g_quit = false;
int g_quitCode = 0;
struct Timer { uintptr_t hwnd; UINT_PTR id; uint64_t interval, due; TIMERPROC proc; };
std::vector<Timer> g_timers;
std::vector<std::pair<HWND, size_t>> g_depth;  // the subclass chain position of each message being handled (DefSubclassProc)

bool g_sdlUp = false;
Uint32 g_wakeType = 0;
std::atomic<bool> g_wakePending{false};
std::unordered_map<SDL_WindowID, uintptr_t> g_bySdl;
HWND g_capture = nullptr, g_active = nullptr;
POINT g_mouse{0, 0};            // the pointer on the screen (pixels), as the last event said
std::bitset<256> g_down;       // the keys down as the last SDL event pumped said (the UI thread's)
uint8_t g_btn = 0;              // MK_LBUTTON | MK_RBUTTON | MK_MBUTTON held
KeySnap g_cur;                  // the state of the input message taken off the queue last
std::atomic<size_t> g_uiThread{0};  // the thread that runs the loop (a hash of its id; 0 until it first calls in)
size_t ThisThread() { return std::hash<std::thread::id>()(std::this_thread::get_id()) | 1; }
void NoteUiThread() { size_t none = 0; g_uiThread.compare_exchange_strong(none, ThisThread()); }
bool OnUiThread() { return g_uiThread.load() == ThisThread(); }
std::function<float(const std::wstring&, int, int)> g_measure;
std::function<bool(const std::string&)>& Hook() { static std::function<bool(const std::string&)> f; return f; }  // a function-local static: set by another file's static initialiser
std::unordered_map<unsigned, SDL_Cursor*> g_cursors;
unsigned g_cursorId = 0;
constexpr UINT_PTR kCaretTimer = 0x7E01;  // an internal timer: the focused text field's caret blinks; never reaches the app

bool EnsureSdl() {
    if (g_sdlUp) return true;
    SDL_SetAppMetadata("WrongGPU", "0.6", "org.wronggpu.WrongGPU");
    SDL_SetHint(SDL_HINT_APP_ID, "WrongGPU");
    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "1");
    SDL_SetHint(SDL_HINT_QUIT_ON_LAST_WINDOW_CLOSE, "0");
    if (!SDL_Init(SDL_INIT_VIDEO)) return false;
    g_wakeType = SDL_RegisterEvents(1);
    g_sdlUp = true;
    return true;
}

Win* FindLocked(HWND h) {
    auto it = g_wins.find((uintptr_t)h);
    return it == g_wins.end() ? nullptr : it->second.get();
}

void Wake() {
    g_cv.notify_all();
    if (g_sdlUp && !g_wakePending.exchange(true)) {
        SDL_Event e;
        SDL_zero(e);
        e.type = g_wakeType;
        SDL_PushEvent(&e);
    }
}

// input: made from an SDL event on the UI thread, so it carries the keys and buttons as that event left them; a post from another
// thread has no pointer position (g_mouse is the UI thread's) and leaves the key state as it was
void Enqueue(HWND h, UINT m, WPARAM wp, LPARAM lp, bool input = false) {
    const bool ui = OnUiThread();
    Queued q{};
    q.msg.hwnd = h; q.msg.message = m; q.msg.wParam = wp; q.msg.lParam = lp; q.msg.time = (DWORD)NowMs();
    q.msg.pt = ui ? g_mouse : POINT{0, 0};
    if (input && ui) { q.hasSnap = true; q.snap.down = g_down; q.snap.btn = g_btn; }
    {
        std::lock_guard<std::mutex> l(g_mu);
        g_q.push_back(q);
    }
    Wake();
}

std::wstring Lower(const std::wstring& s) {
    std::wstring o = s;
    for (auto& c : o) c = (wchar_t)towlower(c);
    return o;
}

HWND Hw(const Win* w) { return (HWND)w->id; }

Win* TopWin(Win* w) {
    while (w && w->parent) w = Find((HWND)w->parent);
    return w;
}

// ---- geometry ---------------------------------------------------------------------------------------------------------------
float DensityGuess(SDL_DisplayID d) {
    // before a window exists: window units are pixels on X11, points (scaled by the display) on Wayland
    const char* drv = SDL_GetCurrentVideoDriver();
    if (drv && !strcmp(drv, "wayland")) { const float s = SDL_GetDisplayContentScale(d); return s > 0 ? s : 1.f; }
    return 1.f;
}

#ifdef WRONGGPU_TESTING
float g_testDensity = 0;  // a test's stand-in for a display scale change the offscreen driver cannot make (0: the real one)
#endif
float PixelDensityOf(SDL_Window* s) {
#ifdef WRONGGPU_TESTING
    if (g_testDensity > 0) return g_testDensity;
#endif
    return SDL_GetWindowPixelDensity(s);
}

void SdlSizePx(Win* w, int& pw, int& ph) {
    pw = ph = 0;
    if (!w->sdl) return;
#ifdef WRONGGPU_TESTING
    if (g_testDensity > 0) {
        SDL_GetWindowSize(w->sdl, &pw, &ph);
        pw = (int)std::lround(pw * g_testDensity); ph = (int)std::lround(ph * g_testDensity);
        return;
    }
#endif
    SDL_GetWindowSizeInPixels(w->sdl, &pw, &ph);
}

POINT OriginOf(Win* w) {  // the client area's corner on the screen
    if (!w) return POINT{0, 0};
    if (w->parent) {
        Win* p = Find((HWND)w->parent);
        POINT o = OriginOf(p);
        return POINT{o.x + w->x, o.y + w->y};
    }
    if (w->sdl) {
        int x = 0, y = 0;
        SDL_GetWindowPosition(w->sdl, &x, &y);
        return POINT{(LONG)std::lround(x * w->density), (LONG)std::lround(y * w->density)};
    }
    return POINT{w->x, w->y};
}

void ClientSize(Win* w, int& cw, int& ch) {
    if (w->sdl) SdlSizePx(w, cw, ch);
    else { cw = w->w; ch = w->h; }
}

void RefreshDensity(Win* w) { if (w->sdl) w->density = std::max(0.5f, PixelDensityOf(w->sdl)); }

UINT DpiOf(Win* w) {
    Win* t = TopWin(w);
    if (t && t->sdl) {
#ifdef WRONGGPU_TESTING
        const float s = g_testDensity > 0 ? g_testDensity : SDL_GetWindowDisplayScale(t->sdl);
#else
        const float s = SDL_GetWindowDisplayScale(t->sdl);
#endif
        if (s > 0.25f) return (UINT)std::lround(96 * s);
    }
    return 96;
}

}  // namespace

Win* Find(HWND h) {
    std::lock_guard<std::mutex> l(g_mu);
    return FindLocked(h);
}
POINT ClientOriginOnScreen(Win* w) { return OriginOf(w); }
HWND TopOf(Win* w) { Win* t = TopWin(w); return t ? Hw(t) : nullptr; }

void InvalidateWin(Win* w) {
    Win* t = TopWin(w);
    if (t) t->invalid = true;
    Wake();
}

namespace {

// ---- sending: the subclass chain, then the class ---------------------------------------------------------------------------------------
LRESULT ClassProc(Win* w, UINT m, WPARAM wp, LPARAM lp) {
    if (w->builtinEdit) return EditClassProc(w, m, wp, lp);
    if (w->proc) return w->proc(Hw(w), m, wp, lp);
    return DefWindowProcW(Hw(w), m, wp, lp);
}

// ---- keys ----------------------------------------------------------------------------------------------------------------------
WPARAM VkOf(SDL_Keycode k) {
    if (k >= 'a' && k <= 'z') return (WPARAM)(k - 'a' + 'A');
    if (k >= '0' && k <= '9') return (WPARAM)k;
    if (k >= SDLK_F1 && k <= SDLK_F12) return (WPARAM)(0x70 + (k - SDLK_F1));
    if (k >= SDLK_KP_1 && k <= SDLK_KP_9) return (WPARAM)(0x61 + (k - SDLK_KP_1));
    switch (k) {
        case SDLK_KP_0: return 0x60;
        case SDLK_RETURN: case SDLK_KP_ENTER: return VK_RETURN;
        case SDLK_ESCAPE: return VK_ESCAPE;
        case SDLK_TAB: return VK_TAB;
        case SDLK_BACKSPACE: return VK_BACK;
        case SDLK_SPACE: return VK_SPACE;
        case SDLK_LEFT: return VK_LEFT;
        case SDLK_RIGHT: return VK_RIGHT;
        case SDLK_UP: return VK_UP;
        case SDLK_DOWN: return VK_DOWN;
        case SDLK_HOME: return VK_HOME;
        case SDLK_END: return VK_END;
        case SDLK_PAGEUP: return VK_PRIOR;
        case SDLK_PAGEDOWN: return VK_NEXT;
        case SDLK_DELETE: return VK_DELETE;
        case SDLK_INSERT: return VK_INSERT;
        case SDLK_LSHIFT: case SDLK_RSHIFT: return VK_SHIFT;
        case SDLK_LCTRL: case SDLK_RCTRL: return VK_CONTROL;
        case SDLK_LALT: case SDLK_RALT: return VK_MENU;
        case SDLK_AC_BACK: return VK_BROWSER_BACK;
        case SDLK_GRAVE: return VK_OEM_3;
        case SDLK_MINUS: return 0xBD;
        case SDLK_EQUALS: return 0xBB;
        case SDLK_COMMA: return 0xBC;
        case SDLK_PERIOD: return 0xBE;
        case SDLK_SLASH: return 0xBF;
        case SDLK_SEMICOLON: return 0xBA;
        case SDLK_APOSTROPHE: return 0xDE;
        case SDLK_LEFTBRACKET: return 0xDB;
        case SDLK_RIGHTBRACKET: return 0xDD;
        case SDLK_BACKSLASH: return 0xDC;
        default: return 0;
    }
}

void SetMods(SDL_Keymod mod) {
    g_down[VK_SHIFT] = (mod & SDL_KMOD_SHIFT) != 0;
    g_down[VK_CONTROL] = (mod & SDL_KMOD_CTRL) != 0;
    g_down[VK_MENU] = (mod & SDL_KMOD_ALT) != 0;
}

UINT KeyStates() {
    return (g_btn & MK_LBUTTON) | (g_btn & MK_RBUTTON) | (g_btn & MK_MBUTTON) | (g_down[VK_SHIFT] ? MK_SHIFT : 0) | (g_down[VK_CONTROL] ? MK_CONTROL : 0);
}

// ---- the window under a point ----------------------------------------------------------------------------------------------------------------
Win* ChildAt(Win* top, int px, int py, int& rx, int& ry) {  // the deepest visible child holding the point (client pixels of top)
    rx = px; ry = py;
    Win* best = top;
    std::vector<Win*> kids;
    {
        std::lock_guard<std::mutex> l(g_mu);
        for (auto& e : g_wins) if (e.second->parent == top->id && e.second->visible) kids.push_back(e.second.get());
    }
    std::sort(kids.begin(), kids.end(), [](Win* a, Win* b) { return a->id > b->id; });  // the newest on top
    for (Win* k : kids)
        if (px >= k->x && py >= k->y && px < k->x + k->w && py < k->y + k->h) { best = k; rx = px - k->x; ry = py - k->y; break; }
    return best;
}

HCURSOR CursorFor(LPCWSTR id) { return (HCURSOR)(uintptr_t)(uint16_t)(uintptr_t)id; }

// ---- SDL events to messages ---------------------------------------------------------------------------------------------------------------

LRESULT HitTest(Win* top, int px, int py) {
    const POINT o = OriginOf(top);
    return Send(Hw(top), WM_NCHITTEST, 0, MAKELPARAM((short)(o.x + px), (short)(o.y + py)));
}

void StateSize(Win* w, bool force) {
    int pw = 0, ph = 0;
    SdlSizePx(w, pw, ph);
    if (!force && pw == w->w && ph == w->h) return;
    w->w = pw; w->h = ph;
    const WPARAM st = w->minimized ? SIZE_MINIMIZED : w->maximized ? SIZE_MAXIMIZED : SIZE_RESTORED;
    Enqueue(Hw(w), WM_SIZE, st, MAKELPARAM((WORD)pw, (WORD)ph));
}

void Activate(Win* top, bool on) {
    if (on) {
        if (g_active == Hw(top)) return;
        g_active = Hw(top);
        Send(Hw(top), WM_NCACTIVATE, TRUE, 0);
        Send(Hw(top), WM_ACTIVATE, WA_ACTIVE, 0);
    } else {
        if (g_active != Hw(top)) return;
        Send(Hw(top), WM_NCACTIVATE, FALSE, 0);
        Send(Hw(top), WM_ACTIVATE, WA_INACTIVE, 0);  // the focus window is still known to it here
        g_active = nullptr;
        if (g_focus) {
            const HWND old = g_focus;
            g_focus = nullptr;
            Send(old, WM_KILLFOCUS, 0, 0);
            ShowTextInput(false, top);
        }
    }
}

void TranslateMouse(const SDL_Event& e) {
    SDL_WindowID wid = 0;
    float fx = 0, fy = 0;
    switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION: wid = e.motion.windowID; fx = e.motion.x; fy = e.motion.y; break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN: case SDL_EVENT_MOUSE_BUTTON_UP: wid = e.button.windowID; fx = e.button.x; fy = e.button.y; break;
        case SDL_EVENT_MOUSE_WHEEL: wid = e.wheel.windowID; fx = e.wheel.mouse_x; fy = e.wheel.mouse_y; break;
        default: return;
    }
    Win* top = nullptr;
    {
        std::lock_guard<std::mutex> l(g_mu);
        auto it = g_bySdl.find(wid);
        if (it != g_bySdl.end()) top = FindLocked((HWND)it->second);
    }
    if (!top) return;
    const int px = (int)std::lround(fx * top->density), py = (int)std::lround(fy * top->density);
    const POINT o = OriginOf(top);
    g_mouse = POINT{o.x + px, o.y + py};
    int rx = px, ry = py;
    Win* target = ChildAt(top, px, py, rx, ry);
    if (g_capture) {  // the capture window gets it all, in its own coordinates
        if (Win* c = Find(g_capture)) {
            const POINT co = OriginOf(c);
            target = c; rx = g_mouse.x - co.x; ry = g_mouse.y - co.y;
        }
    }
    const LRESULT ht = target == top && !g_capture ? HitTest(top, px, py) : HTCLIENT;
    const bool nc = ht != HTCLIENT;
    const LPARAM scr = MAKELPARAM((short)g_mouse.x, (short)g_mouse.y), cli = MAKELPARAM((short)rx, (short)ry);

    if (e.type == SDL_EVENT_MOUSE_WHEEL) {
        const int delta = (int)std::lround(e.wheel.y * WHEEL_DELTA * (e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -1 : 1));
        if (delta) Enqueue(Hw(target), WM_MOUSEWHEEL, MAKEWPARAM((WORD)KeyStates(), (WORD)(short)delta), scr, true);
        return;
    }
    if (e.type == SDL_EVENT_MOUSE_MOTION) {
        // moving between the client area and the caption: the leave messages the app asked for with TrackMouseEvent
        if (nc && top->trackLeave && !g_capture) { top->trackLeave = false; Enqueue(Hw(top), WM_MOUSELEAVE, 0, 0, true); }
        if (!nc && top->trackNcLeave) { top->trackNcLeave = false; Enqueue(Hw(top), WM_NCMOUSELEAVE, 0, 0, true); }
        Enqueue(Hw(target), WM_SETCURSOR, (WPARAM)Hw(target), MAKELPARAM((WORD)ht, WM_MOUSEMOVE), true);
        if (nc) Enqueue(Hw(top), WM_NCMOUSEMOVE, (WPARAM)ht, scr, true);
        else Enqueue(Hw(target), WM_MOUSEMOVE, KeyStates(), cli, true);
        return;
    }
    const bool down = e.type == SDL_EVENT_MOUSE_BUTTON_DOWN;
    const int b = e.button.button;
    const int clicks = e.button.clicks;
    if (b == SDL_BUTTON_LEFT) {
        if (down) g_btn |= MK_LBUTTON; else g_btn &= ~MK_LBUTTON;
        g_down[VK_LBUTTON] = down;
        if (nc) {
            if (down) Enqueue(Hw(top), clicks >= 2 ? WM_NCLBUTTONDBLCLK : WM_NCLBUTTONDOWN, (WPARAM)ht, scr, true);
            else Enqueue(Hw(top), WM_NCLBUTTONUP, (WPARAM)ht, scr, true);
        } else {
            Enqueue(Hw(target), down ? (clicks >= 2 ? WM_LBUTTONDBLCLK : WM_LBUTTONDOWN) : WM_LBUTTONUP, KeyStates(), cli, true);
        }
    } else if (b == SDL_BUTTON_RIGHT) {
        if (down) g_btn |= MK_RBUTTON; else g_btn &= ~MK_RBUTTON;
        if (!nc) Enqueue(Hw(target), down ? WM_RBUTTONDOWN : WM_RBUTTONUP, KeyStates(), cli, true);
    } else if (b == SDL_BUTTON_MIDDLE) {
        if (down) g_btn |= MK_MBUTTON; else g_btn &= ~MK_MBUTTON;
        if (!nc) Enqueue(Hw(target), down ? WM_MBUTTONDOWN : WM_MBUTTONUP, KeyStates(), cli, true);
    } else if (b == SDL_BUTTON_X1 || b == SDL_BUTTON_X2) {
        if (!nc) Enqueue(Hw(target), down ? WM_XBUTTONDOWN : WM_XBUTTONUP, MAKEWPARAM((WORD)KeyStates(), (WORD)(b == SDL_BUTTON_X1 ? XBUTTON1 : XBUTTON2)), cli, true);
    }
}

void TranslateKey(const SDL_Event& e) {
    const bool down = e.type == SDL_EVENT_KEY_DOWN;
    SetMods(e.key.mod);
    const WPARAM vk = VkOf(e.key.key);
    if (!vk) return;
    if (vk != VK_SHIFT && vk != VK_CONTROL && vk != VK_MENU) g_down[vk & 255] = down;
    Win* top = nullptr;
    {
        std::lock_guard<std::mutex> l(g_mu);
        auto it = g_bySdl.find(e.key.windowID);
        if (it != g_bySdl.end()) top = FindLocked((HWND)it->second);
    }
    HWND target = g_focus && Find(g_focus) ? g_focus : top ? Hw(top) : nullptr;
    if (!target) return;
    const bool alt = (e.key.mod & SDL_KMOD_ALT) != 0 || vk == VK_MENU || vk == 0x79;
    LPARAM lp = 1 | ((LPARAM)(e.key.scancode & 0xFF) << 16);
    if (e.key.repeat) lp |= 1 << 30;
    if (alt && (e.key.mod & SDL_KMOD_ALT)) lp |= 1 << 29;
    if (!down) lp |= (1 << 30) | ((LPARAM)1 << 31);
    const UINT m = down ? (alt ? WM_SYSKEYDOWN : WM_KEYDOWN) : (alt ? WM_SYSKEYUP : WM_KEYUP);
    Enqueue(target, m, vk, lp, true);
}

void TranslateText(const SDL_Event& e) {
    HWND target = g_focus && Find(g_focus) ? g_focus : nullptr;
    if (!target) return;
    const char* s = e.text.text;
    while (s && *s) {
        uint32_t c = (unsigned char)*s++;
        int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
        c = extra == 3 ? c & 7 : extra == 2 ? c & 15 : extra == 1 ? c & 31 : c;
        while (extra-- > 0 && *s) c = (c << 6) | ((unsigned char)*s++ & 63);
        Enqueue(target, WM_CHAR, (WPARAM)c, 1, true);
    }
}

void TranslateWindow(const SDL_Event& e) {
    Win* w = nullptr;
    {
        std::lock_guard<std::mutex> l(g_mu);
        auto it = g_bySdl.find(e.window.windowID);
        if (it != g_bySdl.end()) w = FindLocked((HWND)it->second);
    }
    if (!w) return;
    switch (e.type) {
        case SDL_EVENT_WINDOW_CLOSE_REQUESTED: Enqueue(Hw(w), WM_CLOSE, 0, 0); break;
        case SDL_EVENT_WINDOW_RESIZED: case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: RefreshDensity(w); StateSize(w, false); break;
        case SDL_EVENT_WINDOW_MOVED: {
            const POINT o = OriginOf(w);
            Enqueue(Hw(w), WM_MOVE, 0, MAKELPARAM((short)o.x, (short)o.y));
            break;
        }
        case SDL_EVENT_WINDOW_MAXIMIZED: w->maximized = true; w->minimized = false; StateSize(w, true); break;
        case SDL_EVENT_WINDOW_MINIMIZED: w->minimized = true; StateSize(w, true); break;
        case SDL_EVENT_WINDOW_RESTORED: w->maximized = false; w->minimized = false; StateSize(w, true); break;
        case SDL_EVENT_WINDOW_EXPOSED: w->invalid = true; break;
        case SDL_EVENT_WINDOW_FOCUS_GAINED: Activate(w, true); break;
        case SDL_EVENT_WINDOW_FOCUS_LOST: Activate(w, false); break;
        case SDL_EVENT_WINDOW_MOUSE_LEAVE:
            if (w->trackLeave && !g_capture) { w->trackLeave = false; Enqueue(Hw(w), WM_MOUSELEAVE, 0, 0); }
            break;
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED: case SDL_EVENT_WINDOW_DISPLAY_CHANGED: {
            RefreshDensity(w);  // before the rect: SetWindowPos and the mouse turn units to pixels with it
            const POINT o = OriginOf(w);
            int pw = 0, ph = 0;
            SdlSizePx(w, pw, ph);
            w->dpiRect = RECT{o.x, o.y, o.x + pw, o.y + ph};  // handed over by pointer: the window's own, whatever else is queued
            const UINT dpi = DpiOf(w);
            Enqueue(Hw(w), WM_DPICHANGED, MAKEWPARAM((WORD)dpi, (WORD)dpi), (LPARAM)&w->dpiRect);
            break;
        }
        default: break;
    }
}

void Translate(const SDL_Event& e) {
    if (e.type == g_wakeType && g_wakeType) { g_wakePending = false; return; }
    switch (e.type) {
        case SDL_EVENT_MOUSE_MOTION: case SDL_EVENT_MOUSE_BUTTON_DOWN: case SDL_EVENT_MOUSE_BUTTON_UP: case SDL_EVENT_MOUSE_WHEEL: TranslateMouse(e); break;
        case SDL_EVENT_KEY_DOWN: case SDL_EVENT_KEY_UP: TranslateKey(e); break;
        case SDL_EVENT_TEXT_INPUT: TranslateText(e); break;
        case SDL_EVENT_QUIT: {  // a signal (Ctrl+C, SIGTERM): the windows are asked to close, as a close button would
            std::vector<HWND> tops;
            {
                std::lock_guard<std::mutex> l(g_mu);
                for (auto& x : g_wins) if (!x.second->parent && x.second->sdl) tops.push_back(Hw(x.second.get()));
            }
            for (HWND h : tops) Enqueue(h, WM_CLOSE, 0, 0);
            break;
        }
        default:
            if (e.type >= SDL_EVENT_WINDOW_FIRST && e.type <= SDL_EVENT_WINDOW_LAST) TranslateWindow(e);
            break;
    }
}

void PumpSdl(int waitMs) {
    if (!g_sdlUp) return;
    SDL_Event e;
    if (waitMs > 0) {
        if (SDL_WaitEventTimeout(&e, waitMs)) Translate(e);
        else return;
    }
    while (SDL_PollEvent(&e)) Translate(e);
}

// ---- the scripted input of a test build ---------------------------------------------------------------------------------------------------
#ifdef WRONGGPU_TESTING
struct Script {
    bool loaded = false, done = false;
    std::vector<std::string> lines;
    size_t next = 0;
    uint64_t waitUntil = 0;
    float mx = 0, my = 0;  // the pointer, in client DIPs
    int retries = 0;
};
Script g_script;

SDL_WindowID MainSdlId() {
    std::lock_guard<std::mutex> l(g_mu);
    SDL_WindowID best = 0;
    uintptr_t bestId = ~(uintptr_t)0;
    for (auto& e : g_bySdl) if (e.second < bestId) { bestId = e.second; best = e.first; }
    return best;
}

void PushMouse(SDL_EventType t, float x, float y, int button, int clicks) {
    const SDL_WindowID mainId = MainSdlId();
    Win* top = nullptr;
    { std::lock_guard<std::mutex> l(g_mu); auto it = g_bySdl.find(mainId); if (it != g_bySdl.end()) top = FindLocked((HWND)it->second); }
    if (!top) return;
    const float k = DpiOf(top) / 96.f / top->density;  // DIPs to SDL window units
    SDL_Event e;
    SDL_zero(e);
    e.type = t;
    if (t == SDL_EVENT_MOUSE_MOTION) { e.motion.windowID = MainSdlId(); e.motion.x = x * k; e.motion.y = y * k; }
    else if (t == SDL_EVENT_MOUSE_WHEEL) { e.wheel.windowID = MainSdlId(); e.wheel.mouse_x = x * k; e.wheel.mouse_y = y * k; e.wheel.y = (float)button; }
    else { e.button.windowID = MainSdlId(); e.button.x = x * k; e.button.y = y * k; e.button.button = (Uint8)button; e.button.clicks = (Uint8)clicks; e.button.down = t == SDL_EVENT_MOUSE_BUTTON_DOWN; }
    SDL_PushEvent(&e);
}

void PushKey(SDL_Keycode k, SDL_Keymod mod, bool down) {
    SDL_Event e;
    SDL_zero(e);
    e.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
    e.key.windowID = MainSdlId();
    e.key.key = k; e.key.mod = mod; e.key.down = down;
    SDL_PushEvent(&e);
}

SDL_Keycode KeyByName(const std::string& n) {
    static const std::map<std::string, SDL_Keycode> t = {
        {"return", SDLK_RETURN}, {"escape", SDLK_ESCAPE}, {"tab", SDLK_TAB}, {"backspace", SDLK_BACKSPACE}, {"left", SDLK_LEFT}, {"right", SDLK_RIGHT},
        {"up", SDLK_UP}, {"down", SDLK_DOWN}, {"home", SDLK_HOME}, {"end", SDLK_END}, {"delete", SDLK_DELETE}, {"space", SDLK_SPACE},
        {"pagedown", SDLK_PAGEDOWN}, {"pageup", SDLK_PAGEUP}};
    auto it = t.find(n);
    if (it != t.end()) return it->second;
    return n.size() == 1 ? (SDL_Keycode)n[0] : SDLK_UNKNOWN;
}

bool ScriptPending() { return g_script.loaded && !g_script.done && g_script.next < g_script.lines.size(); }

void ScriptClickImpl(float x, float y) {
    g_script.mx = x; g_script.my = y;
    PushMouse(SDL_EVENT_MOUSE_MOTION, x, y, 0, 0);
    PushMouse(SDL_EVENT_MOUSE_BUTTON_DOWN, x, y, SDL_BUTTON_LEFT, 1);
    PushMouse(SDL_EVENT_MOUSE_BUTTON_UP, x, y, SDL_BUTTON_LEFT, 1);
}

// one line per step: wait MS | until WHAT (the hook) | clickname NAME (the hook) | move X Y | click X Y | dclick X Y | wheel N | key NAME | key ctrl+NAME | key shift+NAME | text WORDS | shot FILE | dump FILE | close
bool RunScriptStep() {
    Script& s = g_script;
    if (!s.loaded) {
        s.loaded = true;
        if (const char* f = getenv("WRONGGPU_UI_SCRIPT")) {
            FILE* fp = fopen(f, "r");
            if (!fp) { fprintf(stderr, "wronggpu: cannot read the script %s\n", f); s.done = true; return false; }
            char b[4096];
            while (fgets(b, sizeof b, fp)) {
                std::string l = b;
                while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
                if (!l.empty() && l[0] != '#') s.lines.push_back(l);
            }
            fclose(fp);
        } else s.done = true;
    }
    if (s.done || !g_sdlUp || s.next >= s.lines.size() || NowMs() < s.waitUntil) return false;
    if (MainSdlId() == 0) return false;
    const std::string l = s.lines[s.next++];
    const size_t sp = l.find(' ');
    const std::string cmd = l.substr(0, sp), arg = sp == std::string::npos ? "" : l.substr(sp + 1);
    float x = 0, y = 0;
    if (cmd == "wait") s.waitUntil = NowMs() + (uint64_t)atoi(arg.c_str());
    else if (cmd == "move" && sscanf(arg.c_str(), "%f %f", &x, &y) == 2) { s.mx = x; s.my = y; PushMouse(SDL_EVENT_MOUSE_MOTION, x, y, 0, 0); }
    else if ((cmd == "click" || cmd == "dclick") && sscanf(arg.c_str(), "%f %f", &x, &y) == 2) {
        s.mx = x; s.my = y;
        PushMouse(SDL_EVENT_MOUSE_MOTION, x, y, 0, 0);
        PushMouse(SDL_EVENT_MOUSE_BUTTON_DOWN, x, y, SDL_BUTTON_LEFT, 1);
        PushMouse(SDL_EVENT_MOUSE_BUTTON_UP, x, y, SDL_BUTTON_LEFT, 1);
        if (cmd == "dclick") {
            PushMouse(SDL_EVENT_MOUSE_BUTTON_DOWN, x, y, SDL_BUTTON_LEFT, 2);
            PushMouse(SDL_EVENT_MOUSE_BUTTON_UP, x, y, SDL_BUTTON_LEFT, 2);
        }
        s.waitUntil = NowMs() + 150;
    }
    else if (cmd == "wheel") { PushMouse(SDL_EVENT_MOUSE_WHEEL, s.mx, s.my, atoi(arg.c_str()), 0); s.waitUntil = NowMs() + 100; }
    else if (cmd == "key") {
        SDL_Keymod mod = SDL_KMOD_NONE;
        std::string n = arg;
        for (;;) {
            if (n.rfind("ctrl+", 0) == 0) { mod = (SDL_Keymod)(mod | SDL_KMOD_LCTRL); n = n.substr(5); }
            else if (n.rfind("shift+", 0) == 0) { mod = (SDL_Keymod)(mod | SDL_KMOD_LSHIFT); n = n.substr(6); }
            else if (n.rfind("alt+", 0) == 0) { mod = (SDL_Keymod)(mod | SDL_KMOD_LALT); n = n.substr(4); }
            else break;
        }
        const SDL_Keycode k = KeyByName(n);
        PushKey(k, mod, true);
        PushKey(k, mod, false);
        s.waitUntil = NowMs() + 60;
    }
    else if (cmd == "text") {
        SDL_Event e;
        SDL_zero(e);
        e.type = SDL_EVENT_TEXT_INPUT;
        e.text.windowID = MainSdlId();
        static std::string keep;  // the event's text pointer must outlive the push
        keep = arg;
        e.text.text = keep.c_str();
        SDL_PushEvent(&e);
        s.waitUntil = NowMs() + 100;
    }
    else if ((cmd == "shot" || cmd == "dump" || cmd == "clickname") && Hook()) { Hook()(l); s.waitUntil = NowMs() + 150; }
    else if (cmd == "until" && Hook()) {
        if (!Hook()(l)) {  // not yet: asked again soon
            --s.next;
            s.waitUntil = NowMs() + 100;
            if (++s.retries > 300) { fprintf(stderr, "wronggpu: script condition never came true: %s\n", l.c_str()); ++s.next; s.retries = 0; }
        } else s.retries = 0;
    }
    else if (cmd == "close") {
        SDL_Event e;
        SDL_zero(e);
        e.type = SDL_EVENT_WINDOW_CLOSE_REQUESTED;
        e.window.windowID = MainSdlId();
        SDL_PushEvent(&e);
        s.waitUntil = NowMs() + 200;
    } else fprintf(stderr, "wronggpu: script line not understood: %s\n", l.c_str());
    return true;
}
#else
bool RunScriptStep() { return false; }
bool ScriptPending() { return false; }
#endif

// ---- timers and paints -----------------------------------------------------------------------------------------------------------
constexpr uint64_t kPaintGapMs = 14;  // frames are at most ~70 a second (a swap chain's wait for the display has no counterpart here)

// the message filter of GetMessageW and PeekMessageW: a window (null: any) and a range (0, 0: any)
bool Wanted(HWND fh, UINT lo, UINT hi, HWND h, UINT m) {
    return (!fh || fh == h) && ((lo == 0 && hi == 0) || (m >= lo && m <= hi));
}

// remove: the timer moves on to its next period; a message only looked at (PM_NOREMOVE) stays due
bool TakeTimer(MSG* out, uint64_t now, bool remove, HWND fh, UINT lo, UINT hi) {
    std::lock_guard<std::mutex> l(g_mu);
    for (auto& t : g_timers) {
        if (t.due > now || !Wanted(fh, lo, hi, (HWND)t.hwnd, WM_TIMER)) continue;
        if (remove) t.due = now + t.interval;
        out->hwnd = (HWND)t.hwnd; out->message = WM_TIMER; out->wParam = t.id; out->lParam = (LPARAM)t.proc; out->time = (DWORD)now; out->pt = g_mouse;
        return true;
    }
    return false;
}

bool TakePaint(MSG* out, uint64_t now, HWND fh, UINT lo, UINT hi) {
    std::lock_guard<std::mutex> l(g_mu);
    for (auto& e : g_wins) {
        Win* w = e.second.get();
        if (w->parent || w->messageOnly || !w->visible || !w->invalid || w->minimized) continue;
        if (now < w->lastPaintMs + kPaintGapMs || !Wanted(fh, lo, hi, Hw(w), WM_PAINT)) continue;
        out->hwnd = Hw(w); out->message = WM_PAINT; out->wParam = 0; out->lParam = 0; out->time = (DWORD)now; out->pt = g_mouse;
        return true;
    }
    return false;
}

// the time until something is due (timers, a paint held back by the frame gap): -1 for nothing
int64_t NextDueMs(uint64_t now) {
    int64_t best = -1;
    std::lock_guard<std::mutex> l(g_mu);
    for (auto& t : g_timers) { const int64_t d = std::max<int64_t>(0, (int64_t)t.due - (int64_t)now); if (best < 0 || d < best) best = d; }
    for (auto& e : g_wins) {
        Win* w = e.second.get();
        if (w->parent || w->messageOnly || !w->visible || !w->invalid || w->minimized) continue;
        const int64_t d = std::max<int64_t>(1, (int64_t)(w->lastPaintMs + kPaintGapMs) - (int64_t)now);
        if (best < 0 || d < best) best = d;
    }
    return best;
}

// the first queued message the filter wants; remove: taken off the queue, and the keys it carries become what GetKeyState reports
bool TakeQueued(MSG* out, bool remove, HWND fh, UINT lo, UINT hi) {
    std::lock_guard<std::mutex> l(g_mu);
    for (auto it = g_q.begin(); it != g_q.end(); ++it) {
        if (!Wanted(fh, lo, hi, it->msg.hwnd, it->msg.message)) continue;
        *out = it->msg;
        if (remove) {
            if (it->hasSnap) g_cur = it->snap;
            g_q.erase(it);
        }
        return true;
    }
    return false;
}

// the next message, taking SDL's events, paints and timers in as Windows orders them: posted and input first, then WM_QUIT, then
// WM_PAINT, then WM_TIMER. block: wait for one (GetMessage) or return false (PeekMessage). remove: take it off the queue. fh, lo, hi: the
// window and message range wanted.
bool NextMessage(MSG* out, bool block, bool remove, HWND fh = nullptr, UINT lo = 0, UINT hi = 0) {
    NoteUiThread();
    for (;;) {
        if (TakeQueued(out, remove, fh, lo, hi)) return true;
        PumpSdl(0);
        if (RunScriptStep()) continue;
        if (TakeQueued(out, remove, fh, lo, hi)) return true;
        const uint64_t now = NowMs();
        if (g_quit && !fh && Wanted(nullptr, lo, hi, nullptr, WM_QUIT)) {  // a thread message: for no window in particular
            if (remove) g_quit = false;
            *out = MSG{};
            out->message = WM_QUIT; out->wParam = (WPARAM)g_quitCode;
            return true;
        }
        if (TakePaint(out, now, fh, lo, hi)) return true;
        if (TakeTimer(out, now, remove, fh, lo, hi)) return true;
        if (!block) return false;
        int64_t wait = NextDueMs(now);
        if (ScriptPending()) wait = wait < 0 ? 20 : std::min<int64_t>(wait, 20);  // a script step
        if (wait < 0) wait = 1000;
        if (g_sdlUp) PumpSdl((int)std::max<int64_t>(1, wait));
        else {
            std::unique_lock<std::mutex> l(g_mu);
            g_cv.wait_for(l, std::chrono::milliseconds(std::max<int64_t>(1, wait)), [] { return !g_q.empty() || g_quit; });
        }
    }
}

}  // namespace

// ---- the public entry the Linux Gfx and the test hook use ----------------------------------------------------------------------------------
bool Present(HWND hh, const uint8_t* bgra, int w, int h, int stride) {
    Win* t = Find(hh);
    if (!t || !t->sdl || !bgra) return false;
    SDL_Surface* s = SDL_GetWindowSurface(t->sdl);
    if (!s || s->w != w || s->h != h) return false;
    if (SDL_MUSTLOCK(s) && !SDL_LockSurface(s)) return false;
    const bool ok = SDL_ConvertPixels(w, h, SDL_PIXELFORMAT_ARGB8888, bgra, stride, s->format, s->pixels, s->pitch);
    if (SDL_MUSTLOCK(s)) SDL_UnlockSurface(s);
    if (!ok) return false;
    return SDL_UpdateWindowSurface(t->sdl);
}

void SetIcon(HWND hh, const uint8_t* bgra, int w, int h) {  // straight alpha, BGRA
    Win* t = Find(hh);
    if (!t || !t->sdl || !bgra) return;
    SDL_Surface* s = SDL_CreateSurfaceFrom(w, h, SDL_PIXELFORMAT_ARGB8888, const_cast<uint8_t*>(bgra), w * 4);
    if (!s) return;
    SDL_SetWindowIcon(t->sdl, s);
    SDL_DestroySurface(s);
}

std::wstring ClipGet() {
    static std::wstring held;
    if (!g_sdlUp) return held;
    char* t = SDL_GetClipboardText();
    std::wstring w = t ? wgposix::FromUtf8(t, strlen(t)) : L"";
    SDL_free(t);
    return w;
}
void ClipSet(const std::wstring& text) {
    if (g_sdlUp) SDL_SetClipboardText(wgposix::ToUtf8(text).c_str());
}

void SetEditMeasure(std::function<float(const std::wstring&, int, int)> f) { g_measure = std::move(f); }
float EditMeasure(const std::wstring& text, int fontPx, int upTo) {
    if (g_measure) return g_measure(text, fontPx, upTo);
    return (float)upTo * fontPx * 0.55f;
}
#ifdef WRONGGPU_TESTING
void TestSetPixelDensity(float d) { g_testDensity = d; }
#endif
void SetScriptHook(std::function<bool(const std::string&)> f) { Hook() = std::move(f); }
#ifdef WRONGGPU_TESTING
void ScriptClick(float x, float y) { ScriptClickImpl(x, y); }
#else
void ScriptClick(float, float) {}
#endif

void EditViews(HWND parent, std::vector<EditView>& out) {
    out.clear();
    std::vector<Win*> kids;
    {
        std::lock_guard<std::mutex> l(g_mu);
        for (auto& e : g_wins) if (e.second->parent == (uintptr_t)parent && e.second->builtinEdit && e.second->visible) kids.push_back(e.second.get());
    }
    std::sort(kids.begin(), kids.end(), [](Win* a, Win* b) { return a->id < b->id; });
    for (Win* k : kids) {
        EditView v;
        v.h = Hw(k);
        v.rc = RECT{k->x, k->y, k->x + k->w, k->y + k->h};
        v.text = k->text;
        EditState& s = *k->edit;
        v.selStart = std::min(s.anchor, s.caret); v.selEnd = std::max(s.anchor, s.caret); v.caret = s.caret;
        v.cue = s.cue; v.marginL = s.marginL; v.marginR = s.marginR; v.fontPx = s.fontPx; v.scroll = s.scroll;
        v.focused = g_focus == Hw(k);
        v.caretOn = v.focused && ((NowMs() - s.lastInputMs) % 1060 < 530 || NowMs() - s.lastInputMs < 530);
        out.push_back(std::move(v));
    }
}
void SetEditScroll(HWND h, int scroll) {
    if (Win* w = Find(h)) if (w->edit) w->edit->scroll = scroll;
}

void ShowTextInput(bool on, Win* w) {
    Win* t = TopWin(w);
    if (!g_sdlUp || !t || !t->sdl) return;
    if (on) SDL_StartTextInput(t->sdl); else SDL_StopTextInput(t->sdl);
}

LRESULT Send(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    Win* w = Find(h);
    if (!w) return 0;
    if (w->subs.empty()) return ClassProc(w, m, wp, lp);
    const Sub s = w->subs[0];
    g_depth.push_back({h, 0});
    const LRESULT r = s.proc(h, m, wp, lp, s.id, s.data);
    if (!g_depth.empty()) g_depth.pop_back();
    return r;
}

}  // namespace wgwin

using namespace wgwin;

// ================================================================================================================================
// the Win32 calls
// ================================================================================================================================
ATOM RegisterClassExW(const WNDCLASSEXW* wc) {
    if (!wc || !wc->lpszClassName) return 0;
    NoteUiThread();
    std::lock_guard<std::mutex> l(g_mu);
    g_classes[Lower(wc->lpszClassName)] = *wc;
    return (ATOM)(g_classes.size() + 0xC000);
}

namespace {
SDL_HitTestResult HitTestCallback(SDL_Window*, const SDL_Point* area, void* data) {
    Win* top = Find((HWND)data);
    if (!top) return SDL_HITTEST_NORMAL;
    const LRESULT ht = HitTest(top, (int)std::lround(area->x * top->density), (int)std::lround(area->y * top->density));
    switch (ht) {
        case HTCAPTION: return SDL_HITTEST_DRAGGABLE;
        case HTTOPLEFT: return SDL_HITTEST_RESIZE_TOPLEFT;
        case HTTOP: return SDL_HITTEST_RESIZE_TOP;
        case HTTOPRIGHT: return SDL_HITTEST_RESIZE_TOPRIGHT;
        case HTRIGHT: return SDL_HITTEST_RESIZE_RIGHT;
        case HTBOTTOMRIGHT: return SDL_HITTEST_RESIZE_BOTTOMRIGHT;
        case HTBOTTOM: return SDL_HITTEST_RESIZE_BOTTOM;
        case HTBOTTOMLEFT: return SDL_HITTEST_RESIZE_BOTTOMLEFT;
        case HTLEFT: return SDL_HITTEST_RESIZE_LEFT;
        default: return SDL_HITTEST_NORMAL;  // the client area, and the caption buttons: their clicks come as WM_NC* messages
    }
}

void ApplyMinMax(Win* w) {
    if (!w->sdl) return;
    MINMAXINFO mm{};
    mm.ptMinTrackSize = POINT{1, 1};
    Send(Hw(w), WM_GETMINMAXINFO, 0, (LPARAM)&mm);
    if (mm.ptMinTrackSize.x > 1 || mm.ptMinTrackSize.y > 1)
        SDL_SetWindowMinimumSize(w->sdl, (int)std::lround(mm.ptMinTrackSize.x / w->density), (int)std::lround(mm.ptMinTrackSize.y / w->density));
}
}  // namespace

HWND CreateWindowExW(DWORD exStyle, LPCWSTR cls, LPCWSTR title, DWORD style, int x, int y, int w, int h, HWND parent, HMENU menu, HINSTANCE, LPVOID) {
    NoteUiThread();
    auto win = std::make_unique<Win>();
    const std::wstring lc = cls ? Lower(cls) : L"";
    win->cls = cls ? cls : L"";
    win->style = style; win->exStyle = exStyle;
    if (lc == L"edit") {
        win->builtinEdit = true;
        EditInit(win.get());
        EditSetText(win.get(), title ? title : L"", false);  // the title is the text, with the caret after it, as WM_SETTEXT leaves it
    } else {
        win->text = title ? title : L"";
        std::lock_guard<std::mutex> l(g_mu);
        auto it = g_classes.find(lc);
        if (it == g_classes.end()) { SetLastError(1407); return nullptr; }  // ERROR_CANNOT_FIND_WND_CLASS
        win->proc = it->second.lpfnWndProc;
    }
    win->ctrlId = (LONG_PTR)menu;
    const bool child = (style & WS_CHILD) != 0 && parent && parent != HWND_MESSAGE;
    if (child) {
        if (!Find(parent)) { SetLastError(1400); return nullptr; }
        win->parent = (uintptr_t)parent;
        win->x = x; win->y = y; win->w = w; win->h = h;
        win->visible = (style & WS_VISIBLE) != 0;
    } else if (parent == HWND_MESSAGE) {
        win->messageOnly = true;
    } else {
        if (!EnsureSdl()) { fprintf(stderr, "WrongGPU: cannot open a window: %s\n", SDL_GetError()); exit(3); }
        SDL_WindowFlags f = SDL_WINDOW_HIDDEN | SDL_WINDOW_BORDERLESS | SDL_WINDOW_HIGH_PIXEL_DENSITY;
        if (style & WS_THICKFRAME) f |= SDL_WINDOW_RESIZABLE;
        const std::string t8 = wgposix::ToUtf8(title ? title : L"");
        win->sdl = SDL_CreateWindow(t8.c_str(), std::max(1, w), std::max(1, h), f);
        if (!win->sdl) { fprintf(stderr, "WrongGPU: cannot open a window: %s\n", SDL_GetError()); exit(3); }
        win->sdlId = SDL_GetWindowID(win->sdl);
        win->density = std::max(0.5f, PixelDensityOf(win->sdl));
        // the size asked for is in pixels: set again in SDL's window units now that the density is known
        SDL_SetWindowSize(win->sdl, std::max(1, (int)std::lround(w / win->density)), std::max(1, (int)std::lround(h / win->density)));
        if (x != CW_USEDEFAULT && y != CW_USEDEFAULT) SDL_SetWindowPosition(win->sdl, (int)std::lround(x / win->density), (int)std::lround(y / win->density));
        win->w = w; win->h = h;
    }
    Win* raw = win.get();
    {
        std::lock_guard<std::mutex> l(g_mu);
        raw->id = g_nextId++;
        g_wins[raw->id] = std::move(win);
        if (raw->sdl) g_bySdl[raw->sdlId] = raw->id;
    }
    const HWND hw = Hw(raw);
    if (raw->sdl) SDL_SetWindowHitTest(raw->sdl, HitTestCallback, hw);
    Send(hw, WM_NCCREATE, 0, 0);
    Send(hw, WM_CREATE, 0, 0);
    return hw;
}

BOOL IsWindow(HWND h) { return Find(h) != nullptr; }

BOOL DestroyWindow(HWND h) {
    Win* w = Find(h);
    if (!w) return FALSE;
    std::vector<HWND> kids;
    {
        std::lock_guard<std::mutex> l(g_mu);
        for (auto& e : g_wins) if (e.second->parent == (uintptr_t)h) kids.push_back(Hw(e.second.get()));
    }
    for (HWND k : kids) DestroyWindow(k);
    if (g_capture == h) g_capture = nullptr;
    if (g_focus == h) { g_focus = nullptr; }
    Send(h, WM_DESTROY, 0, 0);
    Send(h, WM_NCDESTROY, 0, 0);
    w = Find(h);
    if (!w) return TRUE;
    if (g_active == h) g_active = nullptr;
    SDL_Window* sdl = w->sdl;
    unsigned sid = w->sdlId;
    {
        std::lock_guard<std::mutex> l(g_mu);
        g_timers.erase(std::remove_if(g_timers.begin(), g_timers.end(), [&](const Timer& t) { return t.hwnd == (uintptr_t)h; }), g_timers.end());
        for (auto it = g_q.begin(); it != g_q.end();) it = it->msg.hwnd == h ? g_q.erase(it) : it + 1;
        g_bySdl.erase(sid);
        g_wins.erase((uintptr_t)h);
    }
    if (sdl) SDL_DestroyWindow(sdl);
    return TRUE;
}

HWND FindWindowExW(HWND parent, HWND after, LPCWSTR cls, LPCWSTR) {
    std::vector<Win*> all;
    {
        std::lock_guard<std::mutex> l(g_mu);
        for (auto& e : g_wins) if (e.second->parent == (uintptr_t)parent) all.push_back(e.second.get());
    }
    std::sort(all.begin(), all.end(), [](Win* a, Win* b) { return a->id < b->id; });
    bool seen = after == nullptr;
    for (Win* w : all) {
        if (!seen) { if (Hw(w) == after) seen = true; continue; }
        if (!cls || Lower(w->cls) == Lower(cls)) return Hw(w);
    }
    return nullptr;
}
HWND FindWindowW(LPCWSTR cls, LPCWSTR title) { return FindWindowExW(nullptr, nullptr, cls, title); }

BOOL ShowWindow(HWND h, int cmd) {
    Win* w = Find(h);
    if (!w) return FALSE;
    const bool was = w->visible;
    if (cmd == SW_HIDE) {
        w->visible = false;
        if (w->sdl) SDL_HideWindow(w->sdl);
        if (g_focus == h && w->parent) { /* a hidden field keeps no keys: the app moves them */ }
    } else if (w->sdl) {
        w->visible = true;
        if (!was) ApplyMinMax(w);
        if (cmd == SW_SHOWMAXIMIZED) { SDL_ShowWindow(w->sdl); SDL_MaximizeWindow(w->sdl); w->maximized = true; }
        else if (cmd == SW_SHOWMINNOACTIVE || cmd == SW_MINIMIZE || cmd == SW_SHOWMINIMIZED || cmd == SW_FORCEMINIMIZE) { SDL_ShowWindow(w->sdl); SDL_MinimizeWindow(w->sdl); w->minimized = true; }
        else if (cmd == SW_RESTORE) { SDL_RestoreWindow(w->sdl); w->minimized = w->maximized = false; SDL_ShowWindow(w->sdl); }
        else { SDL_ShowWindow(w->sdl); if (cmd != SW_SHOWNOACTIVATE && cmd != SW_SHOWNA) SDL_RaiseWindow(w->sdl); }
        w->invalid = true;
        // a display server that sends no focus event (the offscreen driver) still gets an active window, as Windows has
        if (!g_active && !w->minimized && cmd != SW_SHOWNOACTIVATE && cmd != SW_SHOWMINNOACTIVE) Activate(w, true);
        if (!was) Send(h, WM_SHOWWINDOW, TRUE, 0);
    } else {
        w->visible = true;
        if (w->builtinEdit) InvalidateWin(w);
    }
    return was;
}

BOOL UpdateWindow(HWND h) {
    Win* w = Find(h);
    if (!w || w->parent || !w->visible || !w->invalid) return TRUE;
    Send(h, WM_PAINT, 0, 0);
    return TRUE;
}

BOOL IsWindowVisible(HWND h) {
    Win* w = Find(h);
    if (!w) return FALSE;
    for (Win* p = w; p; p = p->parent ? Find((HWND)p->parent) : nullptr) if (!p->visible) return FALSE;
    return TRUE;
}
BOOL IsWindowEnabled(HWND h) { return Find(h) != nullptr; }
BOOL MessageBeep(UINT) { return TRUE; }
BOOL IsIconic(HWND h) { Win* w = Find(h); return w && w->minimized; }
BOOL IsZoomed(HWND h) { Win* w = Find(h); return w && w->maximized; }

BOOL SetWindowPos(HWND h, HWND, int x, int y, int cx, int cy, UINT flags) {
    Win* w = Find(h);
    if (!w) return FALSE;
    if (w->parent) {
        if (!(flags & SWP_NOMOVE)) { w->x = x; w->y = y; }
        if (!(flags & SWP_NOSIZE)) { w->w = cx; w->h = cy; }
        return TRUE;
    }
    if (!w->sdl) return TRUE;
    if (!(flags & SWP_NOSIZE) && cx > 0 && cy > 0) SDL_SetWindowSize(w->sdl, (int)std::lround(cx / w->density), (int)std::lround(cy / w->density));
    if (!(flags & SWP_NOMOVE)) SDL_SetWindowPosition(w->sdl, (int)std::lround(x / w->density), (int)std::lround(y / w->density));
    return TRUE;
}

BOOL GetWindowRect(HWND h, RECT* r) {
    Win* w = Find(h);
    if (!w || !r) return FALSE;
    const POINT o = OriginOf(w);
    int cw = 0, ch = 0;
    ClientSize(w, cw, ch);
    *r = RECT{o.x, o.y, o.x + cw, o.y + ch};
    return TRUE;
}
BOOL GetClientRect(HWND h, RECT* r) {
    Win* w = Find(h);
    if (!w || !r) return FALSE;
    int cw = 0, ch = 0;
    ClientSize(w, cw, ch);
    *r = RECT{0, 0, cw, ch};
    return TRUE;
}
BOOL GetWindowPlacement(HWND h, WINDOWPLACEMENT* wp) {
    Win* w = Find(h);
    if (!w || !wp) return FALSE;
    wp->showCmd = w->minimized ? SW_SHOWMINIMIZED : w->maximized ? SW_SHOWMAXIMIZED : SW_SHOWNORMAL;
    wp->flags = 0;
    GetWindowRect(h, &wp->rcNormalPosition);
    return TRUE;
}

int MapWindowPoints(HWND from, HWND to, POINT* pts, UINT n) {
    const POINT of = from ? OriginOf(Find(from)) : POINT{0, 0}, ot = to ? OriginOf(Find(to)) : POINT{0, 0};
    for (UINT i = 0; i < n; ++i) { pts[i].x += of.x - ot.x; pts[i].y += of.y - ot.y; }
    return 0;
}
BOOL ScreenToClient(HWND h, POINT* p) { MapWindowPoints(nullptr, h, p, 1); return TRUE; }
BOOL ClientToScreen(HWND h, POINT* p) { MapWindowPoints(h, nullptr, p, 1); return TRUE; }
HWND GetParent(HWND h) { Win* w = Find(h); return w ? (HWND)w->parent : nullptr; }

LONG_PTR GetWindowLongPtrW(HWND h, int index) {
    Win* w = Find(h);
    if (!w) return 0;
    return index == GWLP_ID ? w->ctrlId : index == GWLP_USERDATA ? w->userdata : 0;
}
LONG_PTR SetWindowLongPtrW(HWND h, int index, LONG_PTR v) {
    Win* w = Find(h);
    if (!w) return 0;
    LONG_PTR& slot = index == GWLP_ID ? w->ctrlId : w->userdata;
    const LONG_PTR old = slot;
    slot = v;
    return old;
}

BOOL SetWindowTextW(HWND h, LPCWSTR text) { return h ? (BOOL)(Send(h, WM_SETTEXT, 0, (LPARAM)(text ? text : L"")) != 0) : FALSE; }
int GetWindowTextW(HWND h, LPWSTR out, int n) {
    if (!out || n <= 0) return 0;
    return (int)Send(h, WM_GETTEXT, (WPARAM)n, (LPARAM)out);
}
int GetWindowTextLengthW(HWND h) { return (int)Send(h, WM_GETTEXTLENGTH, 0, 0); }

BOOL SetForegroundWindow(HWND h) {
    Win* w = Find(h);
    if (w && w->sdl) SDL_RaiseWindow(w->sdl);
    return TRUE;
}
BOOL FlashWindow(HWND h, BOOL invert) {
    Win* w = Find(h);
    if (w && w->sdl && invert) SDL_FlashWindow(w->sdl, SDL_FLASH_BRIEFLY);
    return TRUE;
}

BOOL InvalidateRect(HWND h, const RECT*, BOOL) {
    Win* w = Find(h);
    if (!w) return FALSE;
    InvalidateWin(w);
    return TRUE;
}
BOOL RedrawWindow(HWND h, const RECT*, void*, UINT flags) {
    Win* w = Find(h);
    if (!w) return FALSE;
    // a text field is drawn inside its window's frame: moved inside that frame's paint, it needs no second one
    if (w->parent) return TRUE;
    InvalidateWin(w);
    if (flags & RDW_UPDATENOW) UpdateWindow(h);
    return TRUE;
}
HDC BeginPaint(HWND h, PAINTSTRUCT* ps) {
    Win* w = Find(h);
    if (w) { w->invalid = false; w->lastPaintMs = NowMs(); }
    if (ps) {
        *ps = PAINTSTRUCT{};
        ps->hdc = (HDC)(uintptr_t)1;
        if (w) { int cw = 0, ch = 0; ClientSize(w, cw, ch); ps->rcPaint = RECT{0, 0, cw, ch}; }
    }
    return (HDC)(uintptr_t)1;
}
BOOL EndPaint(HWND, const PAINTSTRUCT*) { return TRUE; }
BOOL EqualRect(const RECT* a, const RECT* b) { return a->left == b->left && a->top == b->top && a->right == b->right && a->bottom == b->bottom; }

HWND WindowFromPoint(POINT p) {
    std::vector<Win*> tops;
    {
        std::lock_guard<std::mutex> l(g_mu);
        for (auto& e : g_wins) if (!e.second->parent && e.second->sdl && e.second->visible) tops.push_back(e.second.get());
    }
    for (Win* t : tops) {
        const POINT o = OriginOf(t);
        int cw = 0, ch = 0;
        ClientSize(t, cw, ch);
        if (p.x >= o.x && p.y >= o.y && p.x < o.x + cw && p.y < o.y + ch) return Hw(t);
    }
    return nullptr;
}
BOOL GetCursorPos(POINT* p) { if (p) *p = g_mouse; return TRUE; }

BOOL SetWindowSubclass(HWND h, SUBCLASSPROC proc, UINT_PTR id, DWORD_PTR data) {
    Win* w = Find(h);
    if (!w) return FALSE;
    for (auto& s : w->subs) if (s.proc == proc && s.id == id) { s.data = data; return TRUE; }
    w->subs.insert(w->subs.begin(), Sub{proc, id, data});
    return TRUE;
}
LRESULT DefSubclassProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    Win* w = Find(h);
    if (!w) return 0;
    size_t at = 0;
    bool found = false;
    for (size_t i = g_depth.size(); i-- > 0;) if (g_depth[i].first == h) { at = g_depth[i].second; found = true; break; }
    const size_t nx = found ? at + 1 : 0;
    if (nx < w->subs.size()) {
        const Sub s = w->subs[nx];
        g_depth.push_back({h, nx});
        const LRESULT r = s.proc(h, m, wp, lp, s.id, s.data);
        if (!g_depth.empty()) g_depth.pop_back();
        return r;
    }
    return ClassProc(w, m, wp, lp);
}
LRESULT CallWindowProcW(WNDPROC proc, HWND h, UINT m, WPARAM wp, LPARAM lp) { return proc ? proc(h, m, wp, lp) : 0; }

LRESULT DefWindowProcW(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    Win* w = Find(h);
    if (!w) return 0;
    switch (m) {
        case WM_NCHITTEST: {  // a borderless window: the resize border is this layer's (the app's toolbar answers the caption itself)
            if (w->parent || w->messageOnly || !(w->style & WS_THICKFRAME) || w->maximized) return HTCLIENT;
            const POINT o = OriginOf(w);
            const int x = GET_X_LPARAM(lp) - o.x, y = GET_Y_LPARAM(lp) - o.y;
            int cw = 0, ch = 0;
            ClientSize(w, cw, ch);
            const int b = std::max(1, GetSystemMetricsForDpi(SM_CXFRAME, DpiOf(w)));
            const bool l = x < b, r = x >= cw - b, t = y < b, bt = y >= ch - b;
            const int cr = b * 2;  // corners are a little longer than the edges
            if ((l && y < cr) || (t && x < cr)) return HTTOPLEFT;
            if ((r && y < cr) || (t && x >= cw - cr)) return HTTOPRIGHT;
            if ((l && y >= ch - cr) || (bt && x < cr)) return HTBOTTOMLEFT;
            if ((r && y >= ch - cr) || (bt && x >= cw - cr)) return HTBOTTOMRIGHT;
            if (l) return HTLEFT;
            if (r) return HTRIGHT;
            if (bt) return HTBOTTOM;
            if (t) return HTTOP;
            return HTCLIENT;
        }
        case WM_NCACTIVATE: return TRUE;
        case WM_ACTIVATE:
            if (LOWORD(wp) != WA_INACTIVE) {  // the window takes the keys unless one of its fields already has them
                const HWND f = g_focus;
                if (!f || TopOf(Find(f)) != h) SetFocus(h);
            }
            return 0;
        case WM_ERASEBKGND: return 1;
        case WM_PAINT: { PAINTSTRUCT ps; BeginPaint(h, &ps); EndPaint(h, &ps); return 0; }
        case WM_CLOSE: DestroyWindow(h); return 0;
        case WM_SYSCOMMAND: {
            const UINT c = (UINT)(wp & 0xFFF0);
            if (c == SC_CLOSE) { Send(h, WM_CLOSE, 0, 0); return 0; }
            if (!w->sdl) return 0;
            if (c == SC_MINIMIZE) { SDL_MinimizeWindow(w->sdl); w->minimized = true; }
            else if (c == SC_MAXIMIZE) SDL_MaximizeWindow(w->sdl);
            else if (c == SC_RESTORE) { SDL_RestoreWindow(w->sdl); }
            return 0;
        }
        case WM_NCLBUTTONDBLCLK:
            if (wp == HTCAPTION && w->sdl) { if (w->maximized) SDL_RestoreWindow(w->sdl); else SDL_MaximizeWindow(w->sdl); }
            return 0;
        case WM_SYSKEYDOWN:
            if (wp == 0x73) { Send(h, WM_SYSCOMMAND, SC_CLOSE, 0); return 0; }  // Alt+F4
            return 0;
        case WM_MOUSEWHEEL:  // a field passes the wheel on to its window
            if (w->parent) return Send((HWND)w->parent, m, wp, lp);
            return 0;
        case WM_SETICON: return 0;
        default: return 0;
    }
}

// ---- messages and timers ----------------------------------------------------------------------------------------------------
BOOL PostMessageW(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    if (!h) { Enqueue(nullptr, m, wp, lp); return TRUE; }
    {
        std::lock_guard<std::mutex> l(g_mu);
        if (!FindLocked(h)) return FALSE;
    }
    Enqueue(h, m, wp, lp);
    return TRUE;
}
LRESULT SendMessageW(HWND h, UINT m, WPARAM wp, LPARAM lp) { return Send(h, m, wp, lp); }

BOOL GetMessageW(MSG* m, HWND h, UINT lo, UINT hi) {
    NextMessage(m, true, true, h, lo, hi);
    return m->message != WM_QUIT;
}
BOOL PeekMessageW(MSG* m, HWND h, UINT lo, UINT hi, UINT remove) { return NextMessage(m, false, (remove & PM_REMOVE) != 0, h, lo, hi); }
BOOL TranslateMessage(const MSG* m) {
    if (m->message == WM_KEYDOWN && (m->wParam == VK_RETURN || m->wParam == VK_TAB || m->wParam == VK_ESCAPE) && !g_cur.down[VK_MENU])
        Enqueue(m->hwnd, WM_CHAR, m->wParam == VK_RETURN ? 13 : m->wParam == VK_TAB ? 9 : 27, m->lParam);
    return TRUE;
}
LRESULT DispatchMessageW(const MSG* m) {
    if (m->message == WM_TIMER && m->lParam) {  // a timer with its own procedure
        reinterpret_cast<TIMERPROC>(m->lParam)(m->hwnd, WM_TIMER, m->wParam, m->time);
        return 0;
    }
    if (!m->hwnd) return 0;
    if (m->message == WM_TIMER && m->wParam == kCaretTimer) {  // the blink: a frame, nothing for the app
        if (Win* w = Find(m->hwnd)) InvalidateWin(w);
        return 0;
    }
    return Send(m->hwnd, m->message, m->wParam, m->lParam);
}
void PostQuitMessage(int code) { g_quit = true; g_quitCode = code; Wake(); }

DWORD MsgWaitForMultipleObjects(DWORD, const HANDLE*, BOOL, DWORD ms, DWORD) {
    const uint64_t until = NowMs() + (ms == INFINITE ? 3600000u : ms);
    for (;;) {
        {
            std::lock_guard<std::mutex> l(g_mu);
            if (!g_q.empty()) return WAIT_OBJECT_0;
        }
        PumpSdl(0);
        {
            std::lock_guard<std::mutex> l(g_mu);
            if (!g_q.empty()) return WAIT_OBJECT_0;
        }
        const uint64_t now = NowMs();
        if (now >= until) return WAIT_TIMEOUT;
        const int w = (int)std::min<uint64_t>(until - now, 20);
        if (g_sdlUp) PumpSdl(w);
        else { std::unique_lock<std::mutex> l(g_mu); g_cv.wait_for(l, std::chrono::milliseconds(w), [] { return !g_q.empty(); }); }
    }
}

UINT_PTR SetTimer(HWND h, UINT_PTR id, UINT ms, TIMERPROC proc) {
    if (!h) return 0;
    {
        std::lock_guard<std::mutex> l(g_mu);
        const uint64_t now = NowMs();
        for (auto& t : g_timers) if (t.hwnd == (uintptr_t)h && t.id == id) { t.interval = ms; t.due = now + ms; t.proc = proc; Wake(); return id; }
        g_timers.push_back(Timer{(uintptr_t)h, id, ms, now + ms, proc});
    }
    Wake();
    return id;
}
BOOL KillTimer(HWND h, UINT_PTR id) {
    std::lock_guard<std::mutex> l(g_mu);
    const size_t n = g_timers.size();
    g_timers.erase(std::remove_if(g_timers.begin(), g_timers.end(), [&](const Timer& t) { return t.hwnd == (uintptr_t)h && t.id == id; }), g_timers.end());
    return g_timers.size() != n;
}
UINT GetDoubleClickTime() { return 500; }

HWND SetFocus(HWND h) {
    const HWND old = g_focus;
    Win* nw = h ? Find(h) : nullptr;
    if (h && !nw) return old;
    if (old == h) return old;
    g_focus = h;
    if (old && Find(old)) Send(old, WM_KILLFOCUS, (WPARAM)h, 0);
    if (nw) Send(h, WM_SETFOCUS, (WPARAM)old, 0);
    ShowTextInput(nw && nw->builtinEdit, nw ? nw : (old ? Find(old) : nullptr));
    // the blinking caret's clock: only while a field has the keys
    Win* top = nw ? TopWin(nw) : nullptr;
    if (top) {
        if (nw->builtinEdit) SetTimer(Hw(top), kCaretTimer, 530, nullptr); else KillTimer(Hw(top), kCaretTimer);
    }
    return old;
}
HWND GetFocus() { return g_focus && Find(g_focus) ? g_focus : nullptr; }
HWND SetCapture(HWND h) {
    const HWND old = g_capture;
    g_capture = h;
    if (old && old != h && Find(old)) Send(old, WM_CAPTURECHANGED, 0, (LPARAM)h);  // the window that lost it learns who has it
    return old;
}
BOOL ReleaseCapture() {
    const HWND old = g_capture;
    g_capture = nullptr;
    if (old && Find(old)) Send(old, WM_CAPTURECHANGED, 0, 0);
    return TRUE;
}
HWND GetCapture() { return g_capture && Find(g_capture) ? g_capture : nullptr; }
short GetKeyState(int vk) {  // as of the input message taken off the queue last, as Win32's is
    if (vk == VK_LBUTTON) return (g_cur.btn & MK_LBUTTON) ? (short)0x8000 : 0;
    return vk >= 0 && vk < 256 && g_cur.down[vk] ? (short)0x8000 : 0;
}
BOOL TrackMouseEvent(TRACKMOUSEEVENT* t) {
    Win* w = t ? Find(t->hwndTrack) : nullptr;
    if (!w) return FALSE;
    Win* top = TopWin(w);
    if (t->dwFlags & TME_NONCLIENT) top->trackNcLeave = true; else top->trackLeave = true;
    return TRUE;
}
UINT MapVirtualKeyW(UINT code, UINT) { return code; }
int GetKeyNameTextW(LONG, LPWSTR out, int n) { if (out && n > 0) out[0] = 0; return 0; }

// ---- cursors, icons, metrics, monitors ------------------------------------------------------------------------------------------------
HCURSOR LoadCursorW(HINSTANCE, LPCWSTR id) { return CursorFor(id); }
HCURSOR SetCursor(HCURSOR c) {
    const unsigned id = (unsigned)(uintptr_t)c;
    const HCURSOR old = (HCURSOR)(uintptr_t)g_cursorId;
    if (!g_sdlUp || id == g_cursorId) return old;
    g_cursorId = id;
    auto it = g_cursors.find(id);
    if (it == g_cursors.end()) {
        const SDL_SystemCursor k = id == 32649 ? SDL_SYSTEM_CURSOR_POINTER : id == 32513 ? SDL_SYSTEM_CURSOR_TEXT : id == 32650 ? SDL_SYSTEM_CURSOR_PROGRESS : SDL_SYSTEM_CURSOR_DEFAULT;
        it = g_cursors.emplace(id, SDL_CreateSystemCursor(k)).first;
    }
    if (it->second) SDL_SetCursor(it->second);
    return old;
}
HICON LoadIconW(HINSTANCE, LPCWSTR) { return nullptr; }
BOOL DestroyIcon(HICON) { return TRUE; }

int GetSystemMetricsForDpi(int index, UINT dpi) {
    switch (index) {
        case SM_CXFRAME: case SM_CYFRAME: return (int)std::lround(6 * dpi / 96.0);
        case SM_CXPADDEDBORDER: return 0;
        case SM_CXICON: case SM_CYICON: return (int)std::lround(32 * dpi / 96.0);
        case SM_CXSMICON: case SM_CYSMICON: return (int)std::lround(16 * dpi / 96.0);
        default: return 0;
    }
}
int GetSystemMetrics(int index) {
    if (index == SM_CXSCREEN || index == SM_CYSCREEN) {
        MONITORINFO mi{};
        mi.cbSize = sizeof mi;
        GetMonitorInfoW(MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY), &mi);
        return index == SM_CXSCREEN ? mi.rcMonitor.right - mi.rcMonitor.left : mi.rcMonitor.bottom - mi.rcMonitor.top;
    }
    return GetSystemMetricsForDpi(index, GetDpiForSystem());
}
UINT GetDpiForWindow(HWND h) { Win* w = Find(h); return w ? DpiOf(w) : 96; }
UINT GetDpiForSystem() {
    if (!EnsureSdl()) return 96;
    const float s = SDL_GetDisplayContentScale(SDL_GetPrimaryDisplay());
    return s > 0.25f ? (UINT)std::lround(96 * s) : 96;
}
BOOL AdjustWindowRectExForDpi(RECT*, DWORD, BOOL, DWORD, UINT) { return TRUE; }  // no system frame: the client area is the window

BOOL SystemParametersInfoW(UINT action, UINT param, PVOID data, UINT) {
    switch (action) {
        case SPI_GETWHEELSCROLLLINES: if (data) *(UINT*)data = 3; return TRUE;
        case SPI_GETHIGHCONTRAST: if (data) ((HIGHCONTRASTW*)data)->dwFlags = 0; return TRUE;
        case SPI_GETCLIENTAREAANIMATION: if (data) *(BOOL*)data = TRUE; return TRUE;
        case SPI_GETMESSAGEDURATION: if (data) *(UINT*)data = 5; return TRUE;
        default: (void)param; return FALSE;
    }
}
DWORD GetSysColor(int index) {
    switch (index) {
        case COLOR_WINDOW: return RGB(255, 255, 255);
        case COLOR_WINDOWTEXT: return RGB(0, 0, 0);
        case COLOR_HIGHLIGHT: return RGB(0, 120, 215);
        case COLOR_HIGHLIGHTTEXT: return RGB(255, 255, 255);
        case COLOR_GRAYTEXT: return RGB(109, 109, 109);
        default: return 0;
    }
}
HBRUSH GetSysColorBrush(int index) { return CreateSolidBrush(GetSysColor(index)); }

namespace {
SDL_DisplayID DisplayOfPoint(POINT p) {
    SDL_Point sp{p.x, p.y};
    SDL_DisplayID d = SDL_GetDisplayForPoint(&sp);
    return d ? d : SDL_GetPrimaryDisplay();
}
}  // namespace
HMONITOR MonitorFromPoint(POINT p, DWORD) {
    if (!EnsureSdl()) return (HMONITOR)(uintptr_t)1;
    return (HMONITOR)(uintptr_t)DisplayOfPoint(p);
}
HMONITOR MonitorFromRect(const RECT* r, DWORD flags) { return MonitorFromPoint(POINT{r->left, r->top}, flags); }
HMONITOR MonitorFromWindow(HWND h, DWORD flags) {
    Win* w = Find(h);
    if (w && w->sdl) { const SDL_DisplayID d = SDL_GetDisplayForWindow(w->sdl); if (d) return (HMONITOR)(uintptr_t)d; }
    return MonitorFromPoint(POINT{0, 0}, flags);
}
BOOL GetMonitorInfoW(HMONITOR m, MONITORINFO* mi) {
    if (!mi) return FALSE;
    const SDL_DisplayID d = (SDL_DisplayID)(uintptr_t)m;
    SDL_Rect b{0, 0, 1920, 1080}, u{0, 0, 1920, 1080};
    float k = 1;
    if (g_sdlUp && d) { SDL_GetDisplayBounds(d, &b); if (!SDL_GetDisplayUsableBounds(d, &u)) u = b; k = DensityGuess(d); }
    mi->rcMonitor = RECT{(LONG)std::lround(b.x * k), (LONG)std::lround(b.y * k), (LONG)std::lround((b.x + b.w) * k), (LONG)std::lround((b.y + b.h) * k)};
    mi->rcWork = RECT{(LONG)std::lround(u.x * k), (LONG)std::lround(u.y * k), (LONG)std::lround((u.x + u.w) * k), (LONG)std::lround((u.y + u.h) * k)};
    mi->dwFlags = 1;
    return TRUE;
}

HRESULT DwmSetWindowAttribute(HWND, DWORD, LPCVOID, DWORD) { return E_NOTIMPL; }
BOOL ShutdownBlockReasonCreate(HWND, LPCWSTR) { return TRUE; }
BOOL ShutdownBlockReasonDestroy(HWND) { return TRUE; }

HANDLE CreateMutexW(LPSECURITY_ATTRIBUTES, BOOL, LPCWSTR name) {
    std::string dir;
    if (const char* r = getenv("XDG_RUNTIME_DIR"); r && *r && access(r, W_OK) == 0) dir = r;
    else dir = wgposix::XdgDataHome();
    mkdir(dir.c_str(), 0700);
    std::string n = wgposix::ToUtf8(name ? name : L"mutex");
    for (auto& c : n) if (!isalnum((unsigned char)c) && c != '.') c = '_';
    const std::string path = dir + "/" + n + ".lock";
    const int fd = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) { wgposix::SetErrorFromErrno(errno, path); return nullptr; }
    SetLastError(flock(fd, LOCK_EX | LOCK_NB) == 0 ? 0 : ERROR_ALREADY_EXISTS);
    wgposix::Handle* h = wgposix::NewHandle(wgposix::Kind::Mutex);
    h->fd = fd;
    return h;
}

void GetStartupInfoW(STARTUPINFOW* si) { if (si) { memset(si, 0, sizeof *si); si->cb = sizeof *si; } }

INT_PTR ShellExecuteW(HWND, LPCWSTR verb, LPCWSTR file, LPCWSTR params, LPCWSTR dir, int show) {
    SHELLEXECUTEINFOW sei{};
    sei.cbSize = sizeof sei; sei.lpVerb = verb; sei.lpFile = file; sei.lpParameters = params; sei.lpDirectory = dir; sei.nShow = show;
    return ShellExecuteExW(&sei) ? 33 : 31;
}
