// SPDX-License-Identifier: MIT
// Own implementation from public Windows layered-window/GDI API declarations.
#include "hosted_badge_win.h"
#include "overlay.h"
#include "log_win.h"
#include <windows.h>
#include <algorithm>
#include <mutex>

namespace wgwin {
std::wstring HostedBadgeCaption(const wgfsr::UpscaleStatus& status) {
    using wgfsr::UpscaleState;
    if (status.state == UpscaleState::Failed) return L"Upscaling unavailable";
    if (status.state == UpscaleState::StandIn) return L"WrongGPU (stand-in)";
    if (status.state == UpscaleState::Network && status.upscaler == "WrongGPU network") return L"WrongGPU Model M";
    std::wstring label;
    const int chars = MultiByteToWideChar(CP_UTF8, 0, status.upscaler.c_str(), -1, nullptr, 0);
    if (chars > 1) {
        label.resize(size_t(chars));
        MultiByteToWideChar(CP_UTF8, 0, status.upscaler.c_str(), -1, label.data(), chars);
        label.pop_back();
    }
    if (status.state == UpscaleState::Network) return label.empty() ? L"WrongGPU (active)" : label;
    if (label.empty()) return L"AMD FSR";
    if (label.front() >= L'0' && label.front() <= L'9') return L"AMD FSR " + label;
    return label;
}
namespace {
constexpr wchar_t kClass[] = L"WrongGPUHostedProviderBadge";
constexpr COLORREF kTransparent = RGB(1, 2, 3);
std::mutex g_configMutex;
wgfsr::Config g_config;
std::once_flag g_start;
// These are used only on the badge's own message thread.
std::wstring g_caption;
COLORREF g_color = RGB(150, 155, 165);
HFONT g_font = nullptr;
int g_scale = 100;
int Scaled(int n) { return MulDiv(n, g_scale, 100); }

struct FindWindow { HWND badge; HWND game = nullptr; LONG64 area = 0; };
BOOL CALLBACK FindGame(HWND window, LPARAM param) {
    auto& found = *reinterpret_cast<FindWindow*>(param);
    DWORD process = 0;
    GetWindowThreadProcessId(window, &process);
    if (process != GetCurrentProcessId() || window == found.badge || !IsWindowVisible(window) || IsIconic(window) ||
        GetWindow(window, GW_OWNER)) return TRUE;
    RECT r{};
    if (!GetClientRect(window, &r) || r.right < 320 || r.bottom < 200) return TRUE;
    const LONG64 area = LONG64(r.right) * r.bottom;
    if (area > found.area) { found.area = area; found.game = window; }
    return TRUE;
}

LRESULT CALLBACK BadgeProc(HWND window, UINT message, WPARAM wp, LPARAM lp) {
    if (message == WM_NCHITTEST) return HTTRANSPARENT;
    if (message == WM_MOUSEACTIVATE) return MA_NOACTIVATE;
    if (message == WM_ERASEBKGND) return 1;
    if (message == WM_DESTROY) { PostQuitMessage(0); return 0; }
    if (message != WM_PAINT) return DefWindowProcW(window, message, wp, lp);
    PAINTSTRUCT paint{};
    HDC dc = BeginPaint(window, &paint);
    RECT r{};
    GetClientRect(window, &r);
    HBRUSH transparent = CreateSolidBrush(kTransparent);
    FillRect(dc, &r, transparent);
    DeleteObject(transparent);
    HBRUSH card = CreateSolidBrush(RGB(20, 22, 27));
    HPEN edge = CreatePen(PS_SOLID, 1, RGB(55, 60, 68));
    HGDIOBJ oldBrush = SelectObject(dc, card), oldPen = SelectObject(dc, edge);
    RoundRect(dc, 0, 0, r.right, r.bottom, Scaled(14), Scaled(14));
    SelectObject(dc, oldBrush); SelectObject(dc, oldPen);
    DeleteObject(card); DeleteObject(edge);
    HBRUSH dot = CreateSolidBrush(g_color);
    oldBrush = SelectObject(dc, dot); oldPen = SelectObject(dc, GetStockObject(NULL_PEN));
    const int cy = r.bottom / 2;
    Ellipse(dc, Scaled(11), cy-Scaled(4), Scaled(19), cy+Scaled(4));
    SelectObject(dc, oldBrush); SelectObject(dc, oldPen); DeleteObject(dot);
    HGDIOBJ oldFont = SelectObject(dc, g_font);
    SetBkMode(dc, TRANSPARENT); SetTextColor(dc, RGB(242, 244, 248));
    RECT text{Scaled(27), 0, r.right-Scaled(11), r.bottom};
    DrawTextW(dc, g_caption.c_str(), int(g_caption.size()), &text, DT_SINGLELINE|DT_VCENTER|DT_LEFT|DT_NOPREFIX);
    SelectObject(dc, oldFont);
    EndPaint(window, &paint);
    return 0;
}

DWORD WINAPI BadgeThread(void*) {
    HINSTANCE instance = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = BadgeProc; wc.hInstance = instance; wc.lpszClassName = kClass;
    if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return 1;
    HWND badge = CreateWindowExW(WS_EX_LAYERED|WS_EX_TRANSPARENT|WS_EX_NOACTIVATE|WS_EX_TOOLWINDOW,
        kClass, L"WrongGPU provider status", WS_POPUP, 0, 0, 200, 34, nullptr, nullptr, instance, nullptr);
    if (!badge || !SetLayeredWindowAttributes(badge, kTransparent, 255, LWA_COLORKEY|LWA_ALPHA)) {
        if (badge) DestroyWindow(badge);
        wronggpu::Log::Get().Write("hosted badge: Windows badge unavailable (error %lu)", GetLastError());
        return 1;
    }
    bool visible = true, keyDown = false;
    std::wstring previous;
    HWND lastGame = nullptr;
    RECT lastBounds{};
    SetTimer(badge, 1, 100, nullptr);
    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        if (message.message != WM_TIMER || message.hwnd != badge) {
            TranslateMessage(&message); DispatchMessageW(&message); continue;
        }
        wgfsr::Config config;
        { std::lock_guard lock(g_configMutex); config = g_config; }
        FindWindow found{badge}; EnumWindows(FindGame, reinterpret_cast<LPARAM>(&found));
        DWORD foreground = 0; GetWindowThreadProcessId(GetForegroundWindow(), &foreground);
        if (!found.game || foreground != GetCurrentProcessId() || config.overlay == "off") {
            ShowWindow(badge, SW_HIDE); continue;
        }
        const int key = config.overlayKey == "insert" ? VK_END : wgfsr::OverlayKeyCode(config.overlayKey);
        const SHORT keys = key ? GetAsyncKeyState(key) : 0;
        const bool down = (keys & 0x8000) != 0;
        if ((down && !keyDown) || ((keys & 1) && !down && !keyDown)) visible = !visible;
        keyDown = down;
        if (!visible) { ShowWindow(badge, SW_HIDE); continue; }
        wgfsr::UpscaleStatus status;
        const bool active = ReadOverlayProviderStatus(status);
        g_caption = active ? HostedBadgeCaption(status) : L"WrongGPU (idle)";
        g_color = !active ? RGB(150,155,165) : status.state == wgfsr::UpscaleState::Failed ? RGB(255,107,107) : status.state == wgfsr::UpscaleState::Network ? RGB(65,220,125)
                  : status.state == wgfsr::UpscaleState::StandIn ? RGB(245,188,95) : RGB(120,185,255);
        const UINT dpi = GetDpiForWindow(found.game);
        const int scale = dpi ? MulDiv(int(dpi),100,96) : 100;
        if (!g_font || scale != g_scale) {
            if (g_font) DeleteObject(g_font);
            g_scale = scale;
            g_font = CreateFontW(-Scaled(15),0,0,0,FW_SEMIBOLD,FALSE,FALSE,FALSE,DEFAULT_CHARSET,
                OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,L"Segoe UI");
            previous.clear();
        }
        HDC dc = GetDC(badge); HGDIOBJ oldFont = SelectObject(dc,g_font); SIZE size{};
        GetTextExtentPoint32W(dc,g_caption.c_str(),int(g_caption.size()),&size);
        SelectObject(dc,oldFont); ReleaseDC(badge,dc);
        const int width = std::min(int(size.cx)+Scaled(40),Scaled(600)), height = Scaled(34), margin = Scaled(12);
        RECT client{}; POINT origin{};
        if (!GetClientRect(found.game,&client) || !ClientToScreen(found.game,&origin)) { ShowWindow(badge,SW_HIDE); continue; }
        const bool right = config.overlayPosition.find("right") != std::string::npos;
        const bool bottom = config.overlayPosition.find("bottom") != std::string::npos;
        RECT bounds{origin.x+(right ? std::max(margin,int(client.right)-width-margin) : margin),
                    origin.y+(bottom ? std::max(margin,int(client.bottom)-height-margin*4) : margin),0,0};
        bounds.right = bounds.left+width; bounds.bottom = bounds.top+height;
        if (found.game != lastGame) { SetWindowLongPtrW(badge,GWLP_HWNDPARENT,reinterpret_cast<LONG_PTR>(found.game)); lastGame = found.game; }
        if (!IsWindowVisible(badge) || !EqualRect(&bounds,&lastBounds)) {
            SetWindowPos(badge,HWND_TOPMOST,bounds.left,bounds.top,width,height,SWP_NOACTIVATE|SWP_SHOWWINDOW);
            lastBounds = bounds;
        }
        if (g_caption != previous) { previous = g_caption; InvalidateRect(badge,nullptr,FALSE); }
    }
    if (g_font) { DeleteObject(g_font); g_font = nullptr; }
    return 0;
}
}
void ConfigureHostedBadge(const wgfsr::Config& config) { std::lock_guard lock(g_configMutex); g_config = config; }
void ReportHostedBadge() {
    { std::lock_guard lock(g_configMutex); if (g_config.overlay == "off") return; }
    std::call_once(g_start, [] {
        HMODULE pinned = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
                reinterpret_cast<LPCWSTR>(&ReportHostedBadge),&pinned)) return;
        HANDLE thread = CreateThread(nullptr,0,BadgeThread,nullptr,0,nullptr);
        if (thread) CloseHandle(thread);
        else wronggpu::Log::Get().Write("hosted badge: message thread unavailable (error %lu)",GetLastError());
    });
}
}
