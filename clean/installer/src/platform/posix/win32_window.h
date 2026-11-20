// SPDX-License-Identifier: MIT
// Linux: the Win32 window calls the launcher's window makes (CreateWindowExW, the message loop, timers, painting requests,
// focus and capture, cursors, the clipboard, the "EDIT" control) declared the way Windows declares them and implemented on SDL3
// in win32_window_sdl.cpp and win32_window_edit.cpp, so launcher_ui.cpp's WndProc compiles unchanged. A call the layer lacks fails
// the Linux build at that line, as in win32_posix.h.
//
// Conventions:
//  - Coordinates and sizes are device pixels, as Win32's are for a DPI-aware window. A window's DPI is 96 times SDL's display
//    scale for it; the client area is the whole SDL window (there is no system caption: the app's toolbar is its title bar).
//  - One UI thread. PostMessageW (and the timers' wake-up) may come from any thread; every other call is the UI thread's.
//  - The system caption's behaviour comes from WM_NCHITTEST, which the layer asks the window procedure for: HTCAPTION moves the
//    window, the HT*border codes resize it (SDL_SetWindowHitTest), and the caption buttons arrive as WM_NC* messages.
#pragma once
#include <cstdint>
#include <string>

#include "win32_posix.h"

// ---- types -------------------------------------------------------------------------------------------------------------
typedef uintptr_t WPARAM;
typedef intptr_t LPARAM;
typedef intptr_t LRESULT;
typedef uintptr_t UINT_PTR;
typedef intptr_t LONG_PTR;
typedef uint32_t COLORREF;
typedef HANDLE HMENU;
typedef HANDLE HICON;
typedef HANDLE HCURSOR;
typedef HANDLE HFONT;
typedef HANDLE HBRUSH;
typedef HANDLE HBITMAP;
typedef HANDLE HDC;
typedef HANDLE HGDIOBJ;
typedef HANDLE HMONITOR;
typedef void(CALLBACK* TIMERPROC)(HWND, UINT, UINT_PTR, DWORD);
typedef LRESULT(CALLBACK* WNDPROC)(HWND, UINT, WPARAM, LPARAM);
typedef LRESULT(CALLBACK* SUBCLASSPROC)(HWND, UINT, WPARAM, LPARAM, UINT_PTR, DWORD_PTR);
typedef struct tagPOINT { LONG x, y; } POINT, *LPPOINT;
typedef struct tagRECT { LONG left, top, right, bottom; } RECT, *LPRECT;
typedef struct tagSIZE { LONG cx, cy; } SIZE;
typedef struct tagMSG { HWND hwnd; UINT message; WPARAM wParam; LPARAM lParam; DWORD time; POINT pt; } MSG, *LPMSG;
typedef struct tagWNDCLASSEXW {
    UINT cbSize, style; WNDPROC lpfnWndProc; int cbClsExtra, cbWndExtra; HINSTANCE hInstance; HICON hIcon; HCURSOR hCursor; HBRUSH hbrBackground;
    LPCWSTR lpszMenuName, lpszClassName; HICON hIconSm;
} WNDCLASSEXW;
typedef struct tagMINMAXINFO { POINT ptReserved, ptMaxSize, ptMaxPosition, ptMinTrackSize, ptMaxTrackSize; } MINMAXINFO;
typedef struct tagWINDOWPOS { HWND hwnd, hwndInsertAfter; int x, y, cx, cy; UINT flags; } WINDOWPOS;
typedef struct tagNCCALCSIZE_PARAMS { RECT rgrc[3]; WINDOWPOS* lppos; } NCCALCSIZE_PARAMS;
typedef struct tagWINDOWPLACEMENT { UINT length, flags, showCmd; POINT ptMinPosition, ptMaxPosition; RECT rcNormalPosition; } WINDOWPLACEMENT;
typedef struct tagMONITORINFO { DWORD cbSize; RECT rcMonitor, rcWork; DWORD dwFlags; } MONITORINFO;
typedef struct tagPAINTSTRUCT { HDC hdc; BOOL fErase; RECT rcPaint; BOOL fRestore, fIncUpdate; BYTE rgbReserved[32]; } PAINTSTRUCT;
typedef struct tagTRACKMOUSEEVENT { DWORD cbSize, dwFlags; HWND hwndTrack; DWORD dwHoverTime; } TRACKMOUSEEVENT;
typedef struct tagTITLEBARINFOEX { DWORD cbSize; RECT rcTitleBar; DWORD rgstate[6]; RECT rgrect[6]; } TITLEBARINFOEX;
typedef struct tagHIGHCONTRASTW { UINT cbSize; DWORD dwFlags; LPWSTR lpszDefaultScheme; } HIGHCONTRASTW;
typedef struct tagTEXTMETRICW { LONG tmHeight, tmAscent, tmDescent, tmInternalLeading, tmExternalLeading, tmAveCharWidth; } TEXTMETRICW;
#define LF_FACESIZE 32
typedef struct tagLOGFONTW { LONG lfHeight, lfWidth, lfEscapement, lfOrientation, lfWeight; BYTE lfItalic, lfUnderline, lfStrikeOut, lfCharSet, lfOutPrecision, lfClipPrecision, lfQuality, lfPitchAndFamily; WCHAR lfFaceName[LF_FACESIZE]; } LOGFONTW;
typedef struct tagINITCOMMONCONTROLSEX { DWORD dwSize, dwICC; } INITCOMMONCONTROLSEX;

#define CCHILDREN_TITLEBAR 5
#define STATE_SYSTEM_INVISIBLE 0x8000
#define ENDSESSION_CRITICAL 0x40000000
#define RGB(r, g, b) ((COLORREF)(((BYTE)(r)) | ((WORD)((BYTE)(g)) << 8) | (((DWORD)(BYTE)(b)) << 16)))
#define GetRValue(c) ((BYTE)((c) & 0xff))
#define GetGValue(c) ((BYTE)(((c) >> 8) & 0xff))
#define GetBValue(c) ((BYTE)(((c) >> 16) & 0xff))
#define MAKELPARAM(l, h) ((LPARAM)(DWORD)MAKELONG(l, h))
#define MAKEWPARAM(l, h) ((WPARAM)(DWORD)MAKELONG(l, h))
#define GET_X_LPARAM(lp) ((int)(short)LOWORD(lp))
#define GET_Y_LPARAM(lp) ((int)(short)HIWORD(lp))
#define GET_WHEEL_DELTA_WPARAM(wp) ((short)HIWORD(wp))
#define GET_XBUTTON_WPARAM(wp) (HIWORD(wp))
#define GET_KEYSTATE_WPARAM(wp) (LOWORD(wp))
constexpr int WHEEL_DELTA = 120;
constexpr UINT WHEEL_PAGESCROLL = 0xFFFFFFFFu;
constexpr int XBUTTON1 = 1, XBUTTON2 = 2;
#define MAKEINTRESOURCE MAKEINTRESOURCEW
#define GET_APPCOMMAND_LPARAM(lp) ((short)(HIWORD(lp) & ~0xF000))
constexpr int APPCOMMAND_BROWSER_BACKWARD = 1;
constexpr UINT MB_OK = 0;

// ---- messages ----------------------------------------------------------------------------------------------------------
constexpr UINT WM_NULL = 0x0000, WM_CREATE = 0x0001, WM_DESTROY = 0x0002, WM_MOVE = 0x0003, WM_SIZE = 0x0005, WM_ACTIVATE = 0x0006,
               WM_SETFOCUS = 0x0007, WM_KILLFOCUS = 0x0008, WM_SETREDRAW = 0x000B, WM_SETTEXT = 0x000C, WM_GETTEXT = 0x000D,
               WM_GETTEXTLENGTH = 0x000E, WM_PAINT = 0x000F, WM_CLOSE = 0x0010, WM_QUERYENDSESSION = 0x0011, WM_QUIT = 0x0012,
               WM_ERASEBKGND = 0x0014, WM_SYSCOLORCHANGE = 0x0015, WM_ENDSESSION = 0x0016, WM_SHOWWINDOW = 0x0018,
               WM_SETTINGCHANGE = 0x001A, WM_SETCURSOR = 0x0020, WM_GETMINMAXINFO = 0x0024, WM_SETFONT = 0x0030, WM_GETFONT = 0x0031,
               WM_GETOBJECT = 0x003D, WM_NCCREATE = 0x0081, WM_NCDESTROY = 0x0082, WM_NCCALCSIZE = 0x0083, WM_NCHITTEST = 0x0084,
               WM_NCACTIVATE = 0x0086, WM_GETDLGCODE = 0x0087, WM_NCMOUSEMOVE = 0x00A0, WM_NCLBUTTONDOWN = 0x00A1,
               WM_NCLBUTTONUP = 0x00A2, WM_NCLBUTTONDBLCLK = 0x00A3, WM_KEYDOWN = 0x0100, WM_KEYUP = 0x0101, WM_CHAR = 0x0102,
               WM_SYSKEYDOWN = 0x0104, WM_SYSKEYUP = 0x0105, WM_COMMAND = 0x0111, WM_SYSCOMMAND = 0x0112, WM_TIMER = 0x0113,
               WM_CTLCOLOREDIT = 0x0133, WM_MOUSEMOVE = 0x0200, WM_LBUTTONDOWN = 0x0201, WM_LBUTTONUP = 0x0202,
               WM_LBUTTONDBLCLK = 0x0203, WM_RBUTTONDOWN = 0x0204, WM_RBUTTONUP = 0x0205, WM_MBUTTONDOWN = 0x0207,
               WM_MBUTTONUP = 0x0208, WM_MOUSEWHEEL = 0x020A, WM_XBUTTONDOWN = 0x020B, WM_XBUTTONUP = 0x020C,
               WM_CAPTURECHANGED = 0x0215, WM_DPICHANGED = 0x02E0, WM_CUT = 0x0300, WM_COPY = 0x0301, WM_PASTE = 0x0302,
               WM_CLEAR = 0x0303, WM_UNDO = 0x0304, WM_DWMCOLORIZATIONCOLORCHANGED = 0x0320, WM_GETTITLEBARINFOEX = 0x033F,
               WM_APPCOMMAND = 0x0319, WM_APP = 0x8000, WM_SETICON = 0x0080, WM_MOUSELEAVE = 0x02A3, WM_NCMOUSELEAVE = 0x02A2;
constexpr UINT EM_GETSEL = 0x00B0, EM_SETSEL = 0x00B1, EM_GETRECT = 0x00B2, EM_REPLACESEL = 0x00C2, EM_UNDO = 0x00C7, EM_SETMARGINS = 0x00D3,
               EM_SETCUEBANNER = 0x1501, EM_GETCUEBANNER = 0x1502;
constexpr UINT EN_SETFOCUS = 0x0100, EN_KILLFOCUS = 0x0200, EN_CHANGE = 0x0300;
constexpr UINT EC_LEFTMARGIN = 1, EC_RIGHTMARGIN = 2;
constexpr UINT DLGC_WANTARROWS = 1, DLGC_WANTCHARS = 0x80, DLGC_HASSETSEL = 0x08;
constexpr UINT WA_INACTIVE = 0, WA_ACTIVE = 1, WA_CLICKACTIVE = 2;
constexpr UINT SIZE_RESTORED = 0, SIZE_MINIMIZED = 1, SIZE_MAXIMIZED = 2;
constexpr UINT SC_KEYMENU = 0xF100, SC_CLOSE = 0xF060, SC_MINIMIZE = 0xF020, SC_MAXIMIZE = 0xF030, SC_RESTORE = 0xF120;
constexpr UINT ICON_SMALL = 0, ICON_BIG = 1;
constexpr UINT MK_LBUTTON = 1, MK_RBUTTON = 2, MK_SHIFT = 4, MK_CONTROL = 8, MK_MBUTTON = 0x10;
constexpr UINT TME_LEAVE = 2, TME_NONCLIENT = 0x10;
constexpr UINT PM_NOREMOVE = 0, PM_REMOVE = 1;
constexpr UINT QS_ALLINPUT = 0x04FF;
constexpr UINT CS_DBLCLKS = 0x0008;
constexpr UINT ICC_WIN95_CLASSES = 0xFF;
constexpr UINT TTM_SETMAXTIPWIDTH = 0x0418;  // (tooltips are not built on Linux: the window's code names this one)

// hit-test codes (WM_NCHITTEST's result)
constexpr LRESULT HTERROR = -2, HTTRANSPARENT = -1, HTNOWHERE = 0, HTCLIENT = 1, HTCAPTION = 2, HTMINBUTTON = 8, HTMAXBUTTON = 9, HTLEFT = 10,
                  HTRIGHT = 11, HTTOP = 12, HTTOPLEFT = 13, HTTOPRIGHT = 14, HTBOTTOM = 15, HTBOTTOMLEFT = 16, HTBOTTOMRIGHT = 17, HTCLOSE = 20;

// ---- window styles, show commands, positions -----------------------------------------------------------------------------
constexpr DWORD WS_OVERLAPPED = 0, WS_POPUP = 0x80000000u, WS_CHILD = 0x40000000, WS_CLIPCHILDREN = 0x02000000, WS_VISIBLE = 0x10000000,
                WS_TABSTOP = 0x00010000, WS_CAPTION = 0x00C00000, WS_SYSMENU = 0x00080000, WS_THICKFRAME = 0x00040000,
                WS_MINIMIZEBOX = 0x00020000, WS_MAXIMIZEBOX = 0x00010000,
                WS_OVERLAPPEDWINDOW = WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX;
constexpr DWORD WS_EX_TOPMOST = 0x8;
constexpr DWORD ES_AUTOHSCROLL = 0x80;
constexpr int CW_USEDEFAULT = (int)0x80000000;
#define HWND_MESSAGE ((HWND)(intptr_t)-3)
constexpr int SW_RESTORE = 9, SW_SHOWMINIMIZED = 2, SW_SHOWMAXIMIZED = 3, SW_SHOWNA = 8, SW_MINIMIZE = 6, SW_FORCEMINIMIZE = 11, SW_MAXIMIZE = 3;
constexpr UINT SWP_NOSIZE = 1, SWP_NOMOVE = 2, SWP_NOZORDER = 4, SWP_NOREDRAW = 8, SWP_NOACTIVATE = 0x10, SWP_FRAMECHANGED = 0x20,
               SWP_NOCOPYBITS = 0x100;
constexpr UINT RDW_INVALIDATE = 1, RDW_ERASE = 4, RDW_FRAME = 0x400, RDW_UPDATENOW = 0x100;
constexpr UINT WPF_RESTORETOMAXIMIZED = 2;
constexpr DWORD MONITOR_DEFAULTTONULL = 0, MONITOR_DEFAULTTOPRIMARY = 1, MONITOR_DEFAULTTONEAREST = 2;
constexpr int SM_CXSCREEN = 0, SM_CYSCREEN = 1, SM_CXFRAME = 32, SM_CYFRAME = 33, SM_CXICON = 11, SM_CYICON = 12, SM_CXSMICON = 49, SM_CYSMICON = 50,
              SM_CXPADDEDBORDER = 92;
constexpr UINT SPI_GETHIGHCONTRAST = 0x42, SPI_GETWHEELSCROLLLINES = 0x68, SPI_SETHIGHCONTRAST = 0x43, SPI_GETCLIENTAREAANIMATION = 0x1042,
               SPI_GETMESSAGEDURATION = 0x2016;
constexpr DWORD HCF_HIGHCONTRASTON = 1;
constexpr int GWLP_ID = -12, GWLP_USERDATA = -21;
constexpr int COLOR_WINDOW = 5, COLOR_WINDOWTEXT = 8, COLOR_HIGHLIGHT = 13, COLOR_HIGHLIGHTTEXT = 14, COLOR_GRAYTEXT = 17, COLOR_HOTLIGHT = 26;
constexpr int FW_NORMAL = 400;
constexpr BYTE CLEARTYPE_QUALITY = 5, DEFAULT_CHARSET = 1;
constexpr DWORD ERROR_BAD_NETPATH = 53, ERROR_NOT_READY = 21;
constexpr UINT CF_UNICODETEXT = 13;
constexpr UINT GMEM_MOVEABLE = 2;

// cursors (LoadCursorW's ids)
#define IDC_ARROW MAKEINTRESOURCEW(32512)
#define IDC_IBEAM MAKEINTRESOURCEW(32513)
#define IDC_APPSTARTING MAKEINTRESOURCEW(32650)
#define IDC_HAND MAKEINTRESOURCEW(32649)

// virtual keys
constexpr WPARAM VK_LBUTTON = 0x01, VK_BACK = 0x08, VK_TAB = 0x09, VK_RETURN = 0x0D, VK_SHIFT = 0x10, VK_CONTROL = 0x11, VK_MENU = 0x12,
                 VK_ESCAPE = 0x1B, VK_SPACE = 0x20, VK_PRIOR = 0x21, VK_NEXT = 0x22, VK_END = 0x23, VK_HOME = 0x24, VK_LEFT = 0x25,
                 VK_UP = 0x26, VK_RIGHT = 0x27, VK_DOWN = 0x28, VK_INSERT = 0x2D, VK_DELETE = 0x2E, VK_F4 = 0x73, VK_NUMPAD0 = 0x60, VK_NUMPAD9 = 0x69, VK_F10 = 0x79, VK_BROWSER_BACK = 0xA6,
                 VK_OEM_3 = 0xC0;
constexpr UINT MAPVK_VK_TO_VSC_EX = 4;

// ---- windows -------------------------------------------------------------------------------------------------------------
typedef WORD ATOM;
ATOM RegisterClassExW(const WNDCLASSEXW* wc);
HWND CreateWindowExW(DWORD exStyle, LPCWSTR cls, LPCWSTR title, DWORD style, int x, int y, int w, int h, HWND parent, HMENU menu, HINSTANCE inst, LPVOID param);
BOOL DestroyWindow(HWND h);
BOOL IsWindow(HWND h);
HWND FindWindowW(LPCWSTR cls, LPCWSTR title);   // this process's windows only
HWND FindWindowExW(HWND parent, HWND after, LPCWSTR cls, LPCWSTR title);
BOOL ShowWindow(HWND h, int cmd);
BOOL UpdateWindow(HWND h);
BOOL IsWindowVisible(HWND h);
BOOL IsWindowEnabled(HWND h);
BOOL MessageBeep(UINT type);   // silent: no sound is made
BOOL IsIconic(HWND h);
BOOL IsZoomed(HWND h);
BOOL SetWindowPos(HWND h, HWND after, int x, int y, int w, int cy, UINT flags);
BOOL GetWindowRect(HWND h, RECT* r);
BOOL GetClientRect(HWND h, RECT* r);
BOOL GetWindowPlacement(HWND h, WINDOWPLACEMENT* wp);
int MapWindowPoints(HWND from, HWND to, POINT* pts, UINT n);
BOOL ScreenToClient(HWND h, POINT* p);
BOOL ClientToScreen(HWND h, POINT* p);
HWND GetParent(HWND h);
LONG_PTR GetWindowLongPtrW(HWND h, int index);   // GWLP_ID, GWLP_USERDATA
LONG_PTR SetWindowLongPtrW(HWND h, int index, LONG_PTR v);
BOOL SetWindowTextW(HWND h, LPCWSTR text);
int GetWindowTextW(HWND h, LPWSTR out, int n);
int GetWindowTextLengthW(HWND h);
BOOL SetForegroundWindow(HWND h);
BOOL FlashWindow(HWND h, BOOL invert);
BOOL InvalidateRect(HWND h, const RECT* r, BOOL erase);
BOOL RedrawWindow(HWND h, const RECT* r, void* region, UINT flags);
HDC BeginPaint(HWND h, PAINTSTRUCT* ps);
BOOL EndPaint(HWND h, const PAINTSTRUCT* ps);
BOOL EqualRect(const RECT* a, const RECT* b);
HWND WindowFromPoint(POINT p);
BOOL GetCursorPos(POINT* p);
BOOL SetWindowSubclass(HWND h, SUBCLASSPROC proc, UINT_PTR id, DWORD_PTR data);
LRESULT DefSubclassProc(HWND h, UINT m, WPARAM wp, LPARAM lp);
LRESULT DefWindowProcW(HWND h, UINT m, WPARAM wp, LPARAM lp);
LRESULT CallWindowProcW(WNDPROC proc, HWND h, UINT m, WPARAM wp, LPARAM lp);

// ---- messages and timers --------------------------------------------------------------------------------------------------
BOOL PostMessageW(HWND h, UINT m, WPARAM wp, LPARAM lp);   // any thread; FALSE when the window is gone
LRESULT SendMessageW(HWND h, UINT m, WPARAM wp, LPARAM lp);
BOOL GetMessageW(MSG* m, HWND h, UINT lo, UINT hi);        // 0 for WM_QUIT
BOOL PeekMessageW(MSG* m, HWND h, UINT lo, UINT hi, UINT remove);
BOOL TranslateMessage(const MSG* m);
LRESULT DispatchMessageW(const MSG* m);
void PostQuitMessage(int code);
DWORD MsgWaitForMultipleObjects(DWORD n, const HANDLE* handles, BOOL all, DWORD ms, DWORD mask);
UINT_PTR SetTimer(HWND h, UINT_PTR id, UINT ms, TIMERPROC proc);   // proc: called from the message loop instead of WM_TIMER
BOOL KillTimer(HWND h, UINT_PTR id);
UINT GetDoubleClickTime();
HWND SetFocus(HWND h);
HWND GetFocus();
HWND SetCapture(HWND h);
BOOL ReleaseCapture();
HWND GetCapture();
short GetKeyState(int vk);
BOOL TrackMouseEvent(TRACKMOUSEEVENT* t);
UINT MapVirtualKeyW(UINT code, UINT type);
int GetKeyNameTextW(LONG lp, LPWSTR out, int n);

// ---- cursors, icons, metrics, monitors --------------------------------------------------------------------------------------
HCURSOR LoadCursorW(HINSTANCE inst, LPCWSTR id);
HCURSOR SetCursor(HCURSOR c);
HICON LoadIconW(HINSTANCE inst, LPCWSTR id);
BOOL DestroyIcon(HICON i);
int GetSystemMetrics(int index);
int GetSystemMetricsForDpi(int index, UINT dpi);
UINT GetDpiForWindow(HWND h);
UINT GetDpiForSystem();
BOOL AdjustWindowRectExForDpi(RECT* r, DWORD style, BOOL menu, DWORD exStyle, UINT dpi);
BOOL SystemParametersInfoW(UINT action, UINT param, PVOID data, UINT winIni);
DWORD GetSysColor(int index);
HBRUSH GetSysColorBrush(int index);
HMONITOR MonitorFromPoint(POINT p, DWORD flags);
HMONITOR MonitorFromRect(const RECT* r, DWORD flags);
HMONITOR MonitorFromWindow(HWND h, DWORD flags);
BOOL GetMonitorInfoW(HMONITOR m, MONITORINFO* mi);
HRESULT DwmSetWindowAttribute(HWND h, DWORD attr, LPCVOID value, DWORD size);   // no DWM: always fails
BOOL ShutdownBlockReasonCreate(HWND h, LPCWSTR reason);   // no session manager hook: succeeds, does nothing
BOOL ShutdownBlockReasonDestroy(HWND h);
void GetStartupInfoW(STARTUPINFOW* si);
INT_PTR ShellExecuteW(HWND h, LPCWSTR verb, LPCWSTR file, LPCWSTR params, LPCWSTR dir, int show);   // xdg-open for documents and addresses

// ---- GDI stand-ins: only what the text fields' font and colours need ------------------------------------------------------------
HFONT CreateFontIndirectW(const LOGFONTW* lf);
HBRUSH CreateSolidBrush(COLORREF c);
BOOL DeleteObject(HGDIOBJ o);
HDC GetDC(HWND h);
int ReleaseDC(HWND h, HDC dc);
HGDIOBJ SelectObject(HDC dc, HGDIOBJ o);
int GetTextFaceW(HDC dc, int n, LPWSTR out);
BOOL GetTextMetricsW(HDC dc, TEXTMETRICW* tm);
COLORREF SetTextColor(HDC dc, COLORREF c);
COLORREF SetBkColor(HDC dc, COLORREF c);
BOOL InitCommonControlsEx(const INITCOMMONCONTROLSEX*);

// ---- clipboard (text only) --------------------------------------------------------------------------------------------------------
BOOL OpenClipboard(HWND owner);
BOOL EmptyClipboard();
HANDLE SetClipboardData(UINT format, HANDLE mem);
HANDLE GetClipboardData(UINT format);
BOOL CloseClipboard();
HGLOBAL GlobalAlloc(UINT flags, SIZE_T bytes);
LPVOID GlobalLock(HGLOBAL h);
BOOL GlobalUnlock(HGLOBAL h);
HGLOBAL GlobalFree(HGLOBAL h);

// ---- process start and console (main.cpp, cli.cpp) ------------------------------------------------------------------------------------
void WgSetCommandLine(int argc, char** argv);   // main() hands the process's arguments over; GetCommandLineW returns them quoted
LPWSTR GetCommandLineW();
LPWSTR* CommandLineToArgvW(LPCWSTR cmd, int* argc);
BOOL SetDefaultDllDirectories(DWORD flags);
BOOL AttachConsole(DWORD pid);
constexpr DWORD ATTACH_PARENT_PROCESS = 0xFFFFFFFFu;
BOOL WriteConsoleW(HANDLE h, const void* buf, DWORD n, LPDWORD written, LPVOID reserved);
double _wtof(const wchar_t* s);
int MulDiv(int a, int b, int c);

// ---- what the Linux build's own code (ui_gfx_cairo.cpp, the test hook) asks of the layer --------------------------------------------------
#include <functional>
#include <vector>
namespace wgwin {
// The frame (premultiplied BGRA, `stride` bytes a row) shown in the window; false when the window has no surface or another size.
bool Present(HWND h, const uint8_t* bgra, int w, int h2, int stride);
// What a text field shows, for the drawing code to paint after the frame: its rectangle in the parent's client pixels, text,
// selection, the cue banner, margins, the caret's state and the font's pixel height.
struct EditView {
    HWND h = nullptr;
    RECT rc{};
    std::wstring text, cue;
    int selStart = 0, selEnd = 0, caret = 0, marginL = 0, marginR = 0, fontPx = 13, scroll = 0;
    bool focused = false, caretOn = false;
    // x of the character boundary `i` in the text, in pixels from the field's left edge; set by the painter (Measure)
};
// the visible text fields of `parent`, in creation order
void EditViews(HWND parent, std::vector<EditView>& out);
// The painter's text measure (pixels from the text start to the boundary before character `i` of `text`) for the field's font:
// the caret and the selection are placed with it.
// The text field's horizontal scroll, kept by the painter so the caret stays in view.
void SetEditScroll(HWND h, int scroll);
// The window's icon: straight-alpha BGRA pixels.
void SetIcon(HWND h, const uint8_t* bgra, int w, int h2);
void SetEditMeasure(std::function<float(const std::wstring& text, int fontPx, int upTo)> f);
// Test builds only (WRONGGPU_TESTING): WRONGGPU_UI_SCRIPT names a script of synthetic input run by the message loop; its `shot`
// and `dump` lines call this hook with the rest of the line (the Linux app's test hook: ui_live_hook_linux.cpp).
void SetScriptHook(std::function<bool(const std::string& line)> f);
// For the hook: a left click at (x, y) in the window's client DIPs. A script line that starts with `until` is asked again every
// 100 ms until the hook says true (at most 30 s).
void ScriptClick(float x, float y);
}  // namespace wgwin
