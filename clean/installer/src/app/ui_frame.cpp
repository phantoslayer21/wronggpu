// SPDX-License-Identifier: MIT
// WrongGPU launcher window: its frame (launcher_ui.cpp has the window itself): the title bar's colours as DWM draws what is
// left of it, the window's icon drawn from the mark, its place between runs (launcher.ini), and the offscreen frames of the
// ui-dump and ui-render. Written for this project.
#include "app/launcher_window.h"

#include <commctrl.h>
#include <dwmapi.h>
#include <shlobj.h>
#include <shobjidl.h>

namespace wg::app {
using namespace wg::app::lui;

namespace win {

extern UINT dpi;            // the window's DPI (launcher_ui.cpp)
extern RECT normalRect;     // its last normal place (screen pixels)
extern HICON bigIcon, smallIcon;
extern Gfx gfx;

static COLORREF TitleRgb(Col c) { return RGB((int)std::lround(c.r * 255), (int)std::lround(c.g * 255), (int)std::lround(c.b * 255)); }
// The title bar in the theme's colours (Windows 11: caption, border and text colours; Windows 10: light or dark).
void ThemeTitleBar() {
    if (!v.hwnd || !v.live) return;
    const BOOL dark = pal.dark ? TRUE : FALSE;
    const COLORREF caption = TitleRgb(pal.bg), border = TitleRgb(pal.sep), text = TitleRgb(pal.text);
    // the window has no system caption (the toolbar is its title bar: WM_NCCALCSIZE); these colour what Windows still draws
    // of the frame (its border, the window menu, the snap layouts) and a caption if the custom frame is ever off
    DwmSetWindowAttribute(v.hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof dark);
    DwmSetWindowAttribute(v.hwnd, 35 /* DWMWA_CAPTION_COLOR */, &caption, sizeof caption);
    DwmSetWindowAttribute(v.hwnd, 34 /* DWMWA_BORDER_COLOR */, &border, sizeof border);
    DwmSetWindowAttribute(v.hwnd, 36 /* DWMWA_TEXT_COLOR */, &text, sizeof text);
}

// The window's icon (title bar, taskbar, Alt+Tab): the mark the toolbar draws (AppMark), drawn at the icon's size
HICON MarkIcon(int px) {
    Gfx g2;
    std::vector<uint8_t> bgra;
    UINT w = 0, h = 0;
    if (px <= 0 || !g2.Init() || !g2.BeginOffscreen(24, 24, px / 24.f)) return nullptr;
    g2.Clear(Col{0, 0, 0, 0});
    AppMark(g2, Rc{0, 0, 24, 24});
    g2.EndFrame();
    if (!g2.Pixels(bgra, w, h)) return nullptr;
    for (size_t i = 0; i + 3 < bgra.size(); i += 4)  // icons take straight alpha
        if (const uint8_t a = bgra[i + 3]; a && a < 255)
            for (int c = 0; c < 3; ++c) bgra[i + c] = (uint8_t)std::min(255, bgra[i + c] * 255 / a);
    BITMAPV5HEADER bh{};
    bh.bV5Size = sizeof bh; bh.bV5Width = (LONG)w; bh.bV5Height = -(LONG)h; bh.bV5Planes = 1; bh.bV5BitCount = 32; bh.bV5Compression = BI_BITFIELDS;
    bh.bV5RedMask = 0x00ff0000; bh.bV5GreenMask = 0x0000ff00; bh.bV5BlueMask = 0x000000ff; bh.bV5AlphaMask = 0xff000000;
    void* bits = nullptr;
    HDC dc = GetDC(nullptr);
    HBITMAP color = CreateDIBSection(dc, (BITMAPINFO*)&bh, DIB_RGB_COLORS, &bits, nullptr, 0);
    ReleaseDC(nullptr, dc);
    if (!color || !bits) { if (color) DeleteObject(color); return nullptr; }
    memcpy(bits, bgra.data(), bgra.size());
    HBITMAP mask = CreateBitmap((int)w, (int)h, 1, 1, nullptr);
    ICONINFO ii{TRUE, 0, 0, mask, color};
    HICON icon = CreateIconIndirect(&ii);
    DeleteObject(color);
    DeleteObject(mask);
    return icon;
}
void SetWindowIcons() {
    if (!v.hwnd || !v.live) return;
    HICON large = MarkIcon(GetSystemMetricsForDpi(SM_CXICON, dpi)), little = MarkIcon(GetSystemMetricsForDpi(SM_CXSMICON, dpi));
    if (large) SendMessageW(v.hwnd, WM_SETICON, ICON_BIG, (LPARAM)large);
    if (little) SendMessageW(v.hwnd, WM_SETICON, ICON_SMALL, (LPARAM)little);
    if (large && bigIcon) DestroyIcon(bigIcon);
    if (little && smallIcon) DestroyIcon(smallIcon);
    if (large) bigIcon = large;
    if (little) smallIcon = little;
}

// The window's place for the next start: its last normal rectangle (screen pixels), the DPI it had and whether it was
// maximized, as "left,top,right,bottom,dpi,maximized" in launcher.ini. Only into settings that exist (not before the first
// run saved them, not after the launcher uninstalled itself); the file read again first, so only this line changes.
// Nothing said if it fails: the next start centres the window.
void SavePlacement(HWND hwnd) {
    const wstring ini = Join(gl.paths.dataDir, L"launcher.ini");
    if (!IsFile(ini)) return;
    WINDOWPLACEMENT wp{sizeof wp};
    if (!GetWindowPlacement(hwnd, &wp)) return;
    if (!IsZoomed(hwnd) && !IsIconic(hwnd)) GetWindowRect(hwnd, &normalRect);
    if (normalRect.right <= normalRect.left || normalRect.bottom <= normalRect.top) return;
    const bool maxed = IsZoomed(hwnd) || (IsIconic(hwnd) && (wp.flags & WPF_RESTORETOMAXIMIZED));
    wchar_t b[96];
    swprintf(b, 96, L"%ld,%ld,%ld,%ld,%u,%d", normalRect.left, normalRect.top, normalRect.right, normalRect.bottom, GetDpiForWindow(hwnd), maxed ? 1 : 0);
    try {
        GlobalSettings now = LoadGlobal(gl.paths);
        if (now.window == b) return;
        now.window = b;
        SaveGlobal(gl.paths, now);
        gl.global.window = b;
    } catch (const SetupError& e) { LogLine(g.ctx, L"window place not saved: " + e.msg); }
}
// The saved place, if it is still on a monitor: its rectangle scaled to that monitor's DPI (the saved one's size in DIPs),
// kept inside the work area and at least the least size
bool SavedPlacement(RECT& r, HMONITOR& mon, UINT& mdpi, bool& maxed) {
    long l = 0, t = 0, rr = 0, bb = 0; unsigned sdpi = 0; int mx = 0;
    if (gl.global.window.empty() || swscanf(gl.global.window.c_str(), L"%ld,%ld,%ld,%ld,%u,%d", &l, &t, &rr, &bb, &sdpi, &mx) != 6) return false;
    if (rr - l < 50 || bb - t < 50 || sdpi < 48 || sdpi > 960) return false;
    RECT s{l, t, rr, bb};
    mon = MonitorFromRect(&s, MONITOR_DEFAULTTONULL);  // a monitor that is gone (unplugged): centred on the main one instead
    if (!mon) return false;
    mdpi = sdpi;
    using Fn = HRESULT(WINAPI*)(HMONITOR, int, UINT*, UINT*);
    if (HMODULE sh = LoadLibraryExW(L"shcore.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32))
        if (auto f = reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(sh, "GetDpiForMonitor")))) { UINT y = 0; f(mon, 0, &mdpi, &y); }
    MONITORINFO mi{sizeof mi};
    if (!GetMonitorInfoW(mon, &mi)) return false;
    const RECT& w = mi.rcWork;
    LONG ww = MulDiv(rr - l, (int)mdpi, (int)sdpi), wh = MulDiv(bb - t, (int)mdpi, (int)sdpi);
    ww = std::min<LONG>(std::max<LONG>(ww, MulDiv(760, (int)mdpi, 96)), w.right - w.left);
    wh = std::min<LONG>(std::max<LONG>(wh, MulDiv(560, (int)mdpi, 96)), w.bottom - w.top);
    const LONG x = std::min<LONG>(std::max<LONG>(l, w.left), w.right - ww), y = std::min<LONG>(std::max<LONG>(t, w.top), w.bottom - wh);
    r = RECT{x, y, x + ww, y + wh};
    maxed = mx != 0;
    return true;
}

// ---- offscreen frames (ui-dump, ui-render) ------------------------------------------------------------------------------------
bool RenderFrame(float w, float h, float scale, const wstring& png) {
    v.W = w; v.H = h;
    // the first pass measures the page (content height, scroll range); the scroll asked for is kept for the frame
    float want[P_COUNT];
    for (int p = 0; p < P_COUNT; ++p) want[p] = v.scroll[p];
    if (!gfx.BeginOffscreen((int)w, (int)h, scale)) return false;
    v.paintScrolled = false;
    PaintAll(gfx);
    gfx.EndFrame();
    if (v.paintScrolled) want[v.page] = v.target[v.page];  // the page scrolled itself (RevealRrConfigCard): kept, as the window keeps it
    v.paintScrolled = false;
    for (int p = 0; p < P_COUNT; ++p) v.scroll[p] = v.target[p] = want[p];
    if (!png.empty()) { try { EnsureDir(Parent(png)); } catch (const SetupError&) {} }
    if (!gfx.BeginOffscreen((int)w, (int)h, scale)) return false;
    PaintAll(gfx);
    gfx.EndFrame();
    if (png.empty()) return true;
    if (!gfx.BeginOffscreen((int)w, (int)h, scale)) return false;
    PaintAll(gfx);
    gfx.EndFrame();
    return gfx.SavePng(png);
}

// A text field's placeholder while it is empty ("Search games"): painted here in the theme's text2 (4.5:1 or better on the
// fill, as FieldBox draws it offscreen, during a page's entrance and under the toolbar's edge), in the field's own font
// where its typed text starts. The EDIT's cue banner (EM_SETCUEBANNER) ignores WM_CTLCOLOREDIT and draws in a fixed system
// grey: 2.2:1 on the dark fill, and the placeholder changed colour each time the native field took over from the painted one.
COLORREF Ref(Col c);  // launcher_ui.cpp
LRESULT EditCueProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    const wstring* cue = nullptr;
    for (auto* e : {&v.edSearch, &v.edUrl}) if (e->h == h) cue = &e->cue;
    for (auto& e : v.edOpti) if (e.h == h) cue = &e.cue;
    if (m == WM_PAINT && !wp && cue && !cue->empty() && GetWindowTextLengthW(h) == 0) {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);  // the caret hidden meanwhile
        const HBRUSH br = (HBRUSH)SendMessageW(GetParent(h), WM_CTLCOLOREDIT, (WPARAM)dc, (LPARAM)h);  // the fill, as the EDIT has it
        RECT rc;
        GetClientRect(h, &rc);
        FillRect(dc, &rc, br ? br : GetSysColorBrush(COLOR_WINDOW));
        RECT fr = rc;
        SendMessageW(h, EM_GETRECT, 0, (LPARAM)&fr);  // where its text is drawn
        const HGDIOBJ old = SelectObject(dc, (HGDIOBJ)SendMessageW(h, WM_GETFONT, 0, 0));
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, Ref(pal.text2));
        DrawTextW(dc, cue->c_str(), -1, &fr, DT_SINGLELINE | DT_NOPREFIX | DT_END_ELLIPSIS | DT_LEFT | DT_TOP);
        SelectObject(dc, old);
        EndPaint(h, &ps);
        return 0;
    }
    // a change between empty and not (typed, pasted, cut, undone, set): painted again whole, so no placeholder is left
    // beside the first character or missing after the last one is deleted
    const bool edits = cue && (m == WM_CHAR || m == WM_KEYDOWN || m == WM_SETTEXT || m == WM_PASTE || m == WM_CUT || m == WM_CLEAR ||
                               m == WM_UNDO || m == EM_UNDO || m == EM_REPLACESEL);
    const bool wasEmpty = edits && GetWindowTextLengthW(h) == 0;
    const LRESULT r = DefSubclassProc(h, m, wp, lp);
    if (edits && wasEmpty != (GetWindowTextLengthW(h) == 0)) InvalidateRect(h, nullptr, TRUE);
    return r;
}

int Px(float dip);  // DIPs to the window's pixels (launcher_ui.cpp)
// Hover tooltips (v.tips: an icon-only button, a name cut off with an ellipsis): one comctl32 tooltip whose tool is the hit
// under the pointer; nothing while a menu, a dialog or the first run is open.
HWND tipWnd = nullptr;
int tipFor = 0;
RECT tipRect{};
wstring tipText;  // the words the tooltip has now
// The same words for the keyboard: the control the keyboard is on (its ring showing) gets its tooltip too, under it, as
// WinUI's ToolTip shows on keyboard focus: a name cut off with an ellipsis is read in full without a mouse (SHIP-BAR 2). A
// second comctl32 tooltip, tracking (placed by us, kept on screen by comctl32); gone with the ring, a menu, a dialog, the
// first run, another active window, or while the pointer's own tip shows the same control.
HWND keyTipWnd = nullptr;
int keyTipFor = 0;
POINT keyTipAt{};
wstring keyTipText;
void UpdateKeyTip() {
    int want = 0;
    const Hit* on = nullptr;
    const auto it = v.tips.find(v.focus);
    if (it != v.tips.end() && !it->second.empty() && v.focusVisible && !v.menu.open && !v.dialog.open && !v.first.open && v.active &&
        !(tipFor == v.focus && v.hot == v.focus))
        for (auto& h : v.hits)
            if (h.id == v.focus && h.shown) { want = h.id; on = &h; }
    if (want == 0 && keyTipFor == 0) return;
    if (!keyTipWnd) {
        if (!want) return;
        keyTipWnd = CreateWindowExW(WS_EX_TOPMOST, TOOLTIPS_CLASSW, nullptr, WS_POPUP | TTS_NOPREFIX | TTS_ALWAYSTIP, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
                                    CW_USEDEFAULT, v.hwnd, nullptr, g.hinst, nullptr);
        if (!keyTipWnd) return;
        TTTOOLINFOW ti{};
        ti.cbSize = TTTOOLINFOW_V2_SIZE; ti.uFlags = TTF_TRACK; ti.hwnd = v.hwnd; ti.uId = 2; ti.lpszText = const_cast<wchar_t*>(L"");
        SendMessageW(keyTipWnd, TTM_ADDTOOLW, 0, (LPARAM)&ti);
    }
    TTTOOLINFOW ti{};
    ti.cbSize = TTTOOLINFOW_V2_SIZE; ti.hwnd = v.hwnd; ti.uId = 2;
    if (!want) {
        SendMessageW(keyTipWnd, TTM_TRACKACTIVATE, FALSE, (LPARAM)&ti);
        keyTipFor = 0;
        return;
    }
    SendMessageW(keyTipWnd, TTM_SETMAXTIPWIDTH, 0, Px(480));
    // under the control's left edge, 6 DIPs below it (where the pointer's tip would be, not over the ring)
    POINT at{(LONG)std::floor(on->r.x * dpi / 96.f), (LONG)std::ceil((on->r.b() + 6) * dpi / 96.f)};
    ClientToScreen(v.hwnd, &at);
    if (it->second != keyTipText || want != keyTipFor) {
        keyTipText = it->second;
        ti.lpszText = const_cast<wchar_t*>(keyTipText.c_str());
        SendMessageW(keyTipWnd, TTM_UPDATETIPTEXTW, 0, (LPARAM)&ti);
    }
    if (want != keyTipFor || at.x != keyTipAt.x || at.y != keyTipAt.y) SendMessageW(keyTipWnd, TTM_TRACKPOSITION, 0, MAKELPARAM(at.x, at.y));
    if (want != keyTipFor) SendMessageW(keyTipWnd, TTM_TRACKACTIVATE, TRUE, (LPARAM)&ti);
    keyTipFor = want;
    keyTipAt = at;
}
void UpdateHoverTip();
void UpdateTip() {
    if (!v.live || !v.hwnd) return;
    UpdateHoverTip();
    UpdateKeyTip();
}
void UpdateHoverTip() {
    const auto it = v.tips.find(v.hot);
    const int want = it == v.tips.end() || v.menu.open || v.dialog.open || v.first.open || v.dragThumb ? 0 : v.hot;
    if (want == 0 && tipFor == 0) return;
    RECT rect{};
    if (want)
        for (auto& h : v.hits)
            if (h.id == want) rect = RECT{(LONG)std::floor(h.r.x * dpi / 96.f), (LONG)std::floor(h.r.y * dpi / 96.f), (LONG)std::ceil(h.r.r() * dpi / 96.f), (LONG)std::ceil(h.r.b() * dpi / 96.f)};
    if (want == tipFor) {  // the same hit: its place can have changed (the wheel or a key scrolled, the window was resized)
        if (tipWnd && !EqualRect(&rect, &tipRect)) {
            TTTOOLINFOW ti{};
            ti.cbSize = TTTOOLINFOW_V2_SIZE; ti.hwnd = v.hwnd; ti.uId = 1; ti.rect = rect;
            SendMessageW(tipWnd, TTM_NEWTOOLRECTW, 0, (LPARAM)&ti);
            tipRect = rect;
        }
        // and its words ("Waits until WrongGPU finishes updating A." gone once A is done): the tip shown says it now
        if (tipWnd && want && it->second != tipText) {
            tipText = it->second;
            TTTOOLINFOW ti{};
            ti.cbSize = TTTOOLINFOW_V2_SIZE; ti.hwnd = v.hwnd; ti.uId = 1; ti.lpszText = const_cast<wchar_t*>(tipText.c_str());
            SendMessageW(tipWnd, TTM_UPDATETIPTEXTW, 0, (LPARAM)&ti);
        }
        return;
    }
    if (!tipWnd) {
        INITCOMMONCONTROLSEX icc{sizeof icc, ICC_WIN95_CLASSES};
        InitCommonControlsEx(&icc);
        tipWnd = CreateWindowExW(WS_EX_TOPMOST, TOOLTIPS_CLASSW, nullptr, WS_POPUP | TTS_NOPREFIX | TTS_ALWAYSTIP, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
                                 CW_USEDEFAULT, v.hwnd, nullptr, g.hinst, nullptr);
        if (!tipWnd) return;
        TTTOOLINFOW ti{};
        ti.cbSize = TTTOOLINFOW_V2_SIZE; ti.uFlags = TTF_SUBCLASS; ti.hwnd = v.hwnd; ti.uId = 1; ti.lpszText = const_cast<wchar_t*>(L"");
        SendMessageW(tipWnd, TTM_ADDTOOLW, 0, (LPARAM)&ti);
        SendMessageW(tipWnd, TTM_SETMAXTIPWIDTH, 0, Px(480));  // 480 DIPs at any scale (set again when the DPI changes)
    }
    SendMessageW(tipWnd, TTM_POP, 0, 0);
    tipFor = want;
    TTTOOLINFOW ti{};
    ti.cbSize = TTTOOLINFOW_V2_SIZE; ti.hwnd = v.hwnd; ti.uId = 1; ti.rect = rect;
    tipRect = rect;
    SendMessageW(tipWnd, TTM_NEWTOOLRECTW, 0, (LPARAM)&ti);
    tipText = want ? it->second : wstring();
    ti.lpszText = const_cast<wchar_t*>(tipText.c_str());  // the control keeps its own copy
    SendMessageW(tipWnd, TTM_UPDATETIPTEXTW, 0, (LPARAM)&ti);
    SendMessageW(tipWnd, TTM_ACTIVATE, want ? TRUE : FALSE, 0);
}

// ---- the frames while something moves, and the smooth scroll (launcher_ui.cpp draws them: WM_PAINT) ------------------------
// The overlay scroll thumb is fading out (it is drawn on every frame until it is gone)
bool ThumbFading() {
    if (!v.live || v.contentH[v.page] <= v.H + 1) return false;
    const float a = ThumbAlpha();
    return a > 0 && a < 1;
}
// Something moves: the window draws frame after frame, each asked for once the last is presented (paced by the swap chain,
// at the display's rate), until it stops. The Library's own motion (a cover's fade, the scanning skeleton's cross-fade) and
// a progress bar's travel count too; between the skeleton's (or a busy indicator's) steps nothing moves and timer 1 wakes
// the window.
bool Animating() {
    for (int p = 0; p < P_COUNT; ++p) if (std::fabs(v.scroll[p] - v.target[p]) > 0.5f) return true;
    if (MotionPending()) return true;  // entrances, a toast's fade
    // the Library's motion and a progress bar's travel or glide: at the display's rate, as scrolling; not under a sheet's
    // frozen scrim (a new amount, a step's words or a scan's result change its picture and draw it again: BackdropKey)
    if (!PageUnderSheet()) {
        if (LibraryMoving()) return true;
        if (v.live && v.barMoving) return true;
    }
    if (ThumbFading()) return true;
    for (auto& a : v.anim) {
        const int id = a.first;
        const int idx = id - ID_TILE;
        const bool hot = v.hot == id || v.hot == ID_TILEQA + idx || (v.focusVisible && (v.focus == id || v.focus == ID_TILEQA + idx));
        if (std::fabs(a.second - (hot ? 1.f : 0.f)) > 0.01f) return true;
    }
    return false;
}
// A change that starts something moving (the wheel, a hover): a frame is asked for; the frames go on from WM_PAINT. Asking
// again while one is pending changes nothing (a touchpad's stream of wheel messages never holds the frames back).
void Kick() { if (v.live && v.hwnd && Animating()) InvalidateRect(v.hwnd, nullptr, FALSE); }
// One step of the smooth scroll and the hover lifts, by the time since the last frame (QueryPerformanceCounter): the same
// speed at 60 Hz and 144 Hz (the old per-tick steps, 0.28 and 0.3 of the way per 16.7 ms)
ULONGLONG lastAdvance = 0;
void Advance() {
    const ULONGLONG now = NowMs();
    const float dt = lastAdvance && now - lastAdvance < 100 ? (float)std::max<ULONGLONG>(1, now - lastAdvance) : 16.7f;
    lastAdvance = now;
    // Windows' "Animation effects" off (Accessibility > Visual effects): scrolling and hover glows jump to where they go
    const bool ease = MotionOn();
    const float ks = 1 - std::pow(0.72f, dt / 16.7f), ka = 1 - std::pow(0.7f, dt / 16.7f);
    for (int p = 0; p < P_COUNT; ++p) {
        const float d = v.target[p] - v.scroll[p];
        if (d == 0) continue;
        v.scroll[p] = std::fabs(d) < 0.6f || !ease ? v.target[p] : v.scroll[p] + d * ks;
        if (p == v.page) v.scrollSeenAt = now;  // the overlay thumb shows while the page moves
    }
    if (v.hot >= ID_TILE && v.hot < ID_TILE + kTileSpan) v.anim[v.hot];
    if (v.hot >= ID_TILEQA && v.hot < ID_TILEQA + kTileSpan) v.anim[v.hot - kTileSpan];
    for (auto& a : v.anim) {
        const int idx = a.first - ID_TILE;
        const bool hot = v.hot == a.first || v.hot == ID_TILEQA + idx || (v.focusVisible && (v.focus == a.first || v.focus == ID_TILEQA + idx));
        const float t = hot ? 1.f : 0.f;
        a.second += (t - a.second) * ka;
        if (std::fabs(t - a.second) < 0.02f || !ease) a.second = t;
    }
}
// The last wheel, touchpad or key scroll (NowMs): hover follows the content only once it rests (WM_PAINT), as Windows and
// macOS have it, so the covers that pass under a still pointer during a glide do not each start their wipe and lift
ULONGLONG scrollInputAt = 0;
void ScrollBy(float dy) {
    const Page p = v.page;
    v.target[p] = std::min(std::max(0.f, v.target[p] + dy), std::max(0.f, v.contentH[p] - v.H));
    v.scrollSeenAt = scrollInputAt = NowMs();
    Kick();
    Invalidate();  // the thumb shows at once (also at either end, where nothing moves)
}
// A precision touchpad's (or a free-spinning wheel's) small steps: the page follows the fingers one to one, no ease after them.
// A glide still under way (a wheel notch, or a touchpad message whose delta happened to be a whole notch: ScrollBy) goes on
// from the new place: never jumped over in one frame in the middle of a swipe.
void ScrollNow(float dy) {
    const Page p = v.page;
    const float maxScroll = std::max(0.f, v.contentH[p] - v.H), glide = v.target[p] - v.scroll[p];
    v.scroll[p] = std::min(std::max(0.f, v.scroll[p] + dy), maxScroll);
    v.target[p] = std::min(std::max(0.f, v.scroll[p] + glide), maxScroll);
    v.scrollSeenAt = scrollInputAt = NowMs();
    if (std::fabs(v.target[p] - v.scroll[p]) > 0.5f) Kick();
    Invalidate();
}

extern int modalDepth;  // launcher_ui.cpp: the nested modal loops open (a note waits for them)
void ModalClosed();
void UninstallIfWaiting();
// The file and folder pickers (split from launcher_ui.cpp: source size)
bool Browse(int mode, wstring& out) {  // 0 exe, 1 folder, 2 model dll, 3 OptiScaler archive, 4 OptiScaler folder, 5 RR model dll
    if (!v.live) return false;
    const bool folder = mode == 1 || mode == 4;
    IFileOpenDialog* d = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog, (void**)&d))) return false;
    DWORD o = 0; d->GetOptions(&o); d->SetOptions(o | FOS_FORCEFILESYSTEM | (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
    if (mode == 1) d->SetTitle(L"Choose a folder that holds games");
    else if (mode == 4) d->SetTitle(L"Choose your unpacked OptiScaler folder");
    else if (mode == 2) { COMDLG_FILTERSPEC spec[] = {{L"DLSS model file (nvngx_dlss.dll)", L"nvngx_dlss.dll"}, {L"DLL files (*.dll)", L"*.dll"}}; d->SetFileTypes(2, spec); d->SetTitle(L"Choose a DLSS file (version 310.5 or newer)"); }
    else if (mode == 5) { COMDLG_FILTERSPEC spec[] = {{L"DLSS Ray Reconstruction file (nvngx_dlssd.dll)", L"nvngx_dlssd.dll"}, {L"DLL files (*.dll)", L"*.dll"}}; d->SetFileTypes(2, spec); d->SetTitle(L"Choose a game's DLSS Ray Reconstruction file"); }
    else if (mode == 3) { COMDLG_FILTERSPEC spec[] = {{L"OptiScaler release (*.7z, *.zip) or its OptiScaler.dll", L"*.7z;*.zip;OptiScaler.dll"}}; d->SetFileTypes(1, spec); d->SetTitle(L"Choose your OptiScaler download"); }
    else {
        COMDLG_FILTERSPEC spec[] = {{L"Game program (*.exe)", L"*.exe"}}; d->SetFileTypes(1, spec); d->SetTitle(L"Choose the game's program");
        // where games are (not the launcher's own folder): a library folder the Library knows, else Program Files; once the
        // player has picked a game the dialog remembers that place itself (a default folder is only its first place)
        // (a folder on a local fixed drive, told by its drive letter alone: no game folder is read on the window's thread, so
        // a sleeping disk or a dropped network share never holds the window up before the picker shows)
        wstring start;
        for (const auto& e : v.games) {
            const wstring p = Parent(e.dir);
            if (start.empty() && e.source != L"Added" && p.size() >= 3 && p[1] == L':' && GetDriveTypeW(p.substr(0, 3).c_str()) == DRIVE_FIXED) start = p;
        }
        if (start.empty()) { PWSTR pf = nullptr; if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_ProgramFilesX86, 0, nullptr, &pf)) && pf) start = pf; if (pf) CoTaskMemFree(pf); }
        IShellItem* place = nullptr;
        if (!start.empty() && SUCCEEDED(SHCreateItemFromParsingName(start.c_str(), nullptr, IID_PPV_ARGS(&place)))) { d->SetDefaultFolder(place); place->Release(); }
    }
    bool ok = false;
    ++modalDepth;  // the file dialog runs its own loop: a note that arrives meanwhile waits for it (InformLater)
    const HRESULT shown = d->Show(v.hwnd);
    --modalDepth;
    ModalClosed();
    UninstallIfWaiting();
    if (SUCCEEDED(shown)) {
        IShellItem* it = nullptr;
        if (SUCCEEDED(d->GetResult(&it))) { PWSTR p = nullptr; if (SUCCEEDED(it->GetDisplayName(SIGDN_FILESYSPATH, &p))) { out = p; CoTaskMemFree(p); ok = true; } it->Release(); }
    }
    d->Release();
    return ok;
}

}  // namespace win
}  // namespace wg::app
