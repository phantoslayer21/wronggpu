// SPDX-License-Identifier: MIT
// WrongGPU UI kit (wg_ui): the owner-drawn widgets. Moved from setup.cpp. Written for this project.
#include "wg_ui/kit.h"

#include <commctrl.h>
#include <dwmapi.h>
#include <objbase.h>

#include <vector>

namespace wg::ui {
using std::wstring;

namespace {
Kit* KitOf(HWND h) { return static_cast<Kit*>(GetPropW(GetAncestor(h, GA_ROOT), L"wgKit")); }
}  // namespace

int Kit::S(int v) const { return (int)(v * scale + 0.5); }
float Kit::SF(float v) const { return (float)(v * scale); }
HBRUSH Kit::Brush(COLORREF c) {
    auto it = brushes.find(c);
    if (it == brushes.end()) it = brushes.emplace(c, CreateSolidBrush(c)).first;
    return it->second;
}
COLORREF Kit::BgOf(HWND h) const { auto it = bg.find(h); return it == bg.end() ? kBg : it->second; }
void Kit::SetBg(HWND h, COLORREF c) { bg[h] = c; }
Gdiplus::Color GC(COLORREF c, BYTE a) { return Gdiplus::Color(a, GetRValue(c), GetGValue(c), GetBValue(c)); }

// Rounded rectangle path (pixels).
void RoundPath(Gdiplus::GraphicsPath& p, float x, float y, float w, float h, float r) {
    r = std::min(r, std::min(w, h) / 2);
    const float d = r * 2;
    if (d <= 0) { p.AddRectangle(Gdiplus::RectF(x, y, w, h)); return; }
    p.AddArc(x, y, d, d, 180, 90);
    p.AddArc(x + w - d, y, d, d, 270, 90);
    p.AddArc(x + w - d, y + h - d, d, d, 0, 90);
    p.AddArc(x, y + h - d, d, d, 90, 90);
    p.CloseFigure();
}
void FillRound(Gdiplus::Graphics& gr, float x, float y, float w, float h, float r, const Gdiplus::Color& c) {
    Gdiplus::GraphicsPath p; RoundPath(p, x, y, w, h, r);
    Gdiplus::SolidBrush b(c); gr.FillPath(&b, &p);
}
void StrokeRound(Gdiplus::Graphics& gr, float x, float y, float w, float h, float r, const Gdiplus::Color& c, float width) {
    Gdiplus::GraphicsPath p; RoundPath(p, x + width / 2, y + width / 2, w - width, h - width, r);
    Gdiplus::Pen pen(c, width); gr.DrawPath(&pen, &p);
}
void Kit::Card(Gdiplus::Graphics& gr, int x, int y, int w, int h, COLORREF fill, COLORREF edge) const {
    FillRound(gr, SF((float)x), SF((float)y), SF((float)w), SF((float)h), SF(10), GC(fill));
    StrokeRound(gr, SF((float)x), SF((float)y), SF((float)w), SF((float)h), SF(10), GC(edge));
}
// A check mark inside the box (pixels).
void CheckMark(Gdiplus::Graphics& gr, float x, float y, float s, const Gdiplus::Color& c, float width) {
    Gdiplus::Pen pen(c, width); pen.SetLineCap(Gdiplus::LineCapRound, Gdiplus::LineCapRound, Gdiplus::DashCapRound);
    pen.SetLineJoin(Gdiplus::LineJoinRound);
    Gdiplus::PointF pts[3] = {{x + s * 0.24f, y + s * 0.52f}, {x + s * 0.43f, y + s * 0.70f}, {x + s * 0.76f, y + s * 0.33f}};
    gr.DrawLines(&pen, pts, 3);
}
void CrossMark(Gdiplus::Graphics& gr, float x, float y, float s, const Gdiplus::Color& c, float width) {
    Gdiplus::Pen pen(c, width); pen.SetLineCap(Gdiplus::LineCapRound, Gdiplus::LineCapRound, Gdiplus::DashCapRound);
    gr.DrawLine(&pen, x + s * 0.32f, y + s * 0.32f, x + s * 0.68f, y + s * 0.68f);
    gr.DrawLine(&pen, x + s * 0.68f, y + s * 0.32f, x + s * 0.32f, y + s * 0.68f);
}
void Kit::UiText(HDC dc, const wstring& t, int x, int y, int w, int h, HFONT f, COLORREF c, UINT fmt) const {
    RECT r{S(x), S(y), S(x + w), S(y + h)};
    HGDIOBJ old = SelectObject(dc, f);
    SetTextColor(dc, c); SetBkMode(dc, TRANSPARENT);
    DrawTextW(dc, t.c_str(), (int)t.size(), &r, fmt);
    SelectObject(dc, old);
}

// ------------------------------------------------------------------------------------------------
// Owner-drawn controls: buttons (BS_OWNERDRAW, painted in WM_DRAWITEM; hover tracked by a subclass), a check box
// class and a progress bar class.

LRESULT CALLBACK ButtonSubclass(HWND h, UINT m, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR ref) {
    Kit& kit = *reinterpret_cast<Kit*>(ref);
    switch (m) {
        case WM_MOUSEMOVE:
            if (kit.hover != h) {
                HWND old = kit.hover; kit.hover = h;
                if (old) InvalidateRect(old, nullptr, FALSE);
                InvalidateRect(h, nullptr, FALSE);
                TRACKMOUSEEVENT t{sizeof(t), TME_LEAVE, h, 0}; TrackMouseEvent(&t);
            }
            break;
        case WM_MOUSELEAVE: if (kit.hover == h) { kit.hover = nullptr; InvalidateRect(h, nullptr, FALSE); } break;
        case WM_SETCURSOR: if (IsWindowEnabled(h)) { SetCursor(LoadCursorW(nullptr, IDC_HAND)); return TRUE; } break;
        case WM_ERASEBKGND: return 1;
        case WM_ENABLE: InvalidateRect(h, nullptr, FALSE); break;
    }
    return DefSubclassProc(h, m, wp, lp);
}

void Kit::DrawButton(const DRAWITEMSTRUCT* d) {
    const int kind = buttonKind.count(d->hwndItem) ? buttonKind[d->hwndItem] : 1;
    const int w = d->rcItem.right - d->rcItem.left, h = d->rcItem.bottom - d->rcItem.top;
    HDC mem = CreateCompatibleDC(d->hDC); HBITMAP bmp = CreateCompatibleBitmap(d->hDC, w, h); HGDIOBJ oldBmp = SelectObject(mem, bmp);
    RECT all{0, 0, w, h}; FillRect(mem, &all, Brush(BgOf(d->hwndItem)));
    const bool disabled = (d->itemState & ODS_DISABLED) != 0, pressed = (d->itemState & ODS_SELECTED) != 0, hot = hover == d->hwndItem && !disabled;
    COLORREF fill = 0, edge = 0, ink = kText; bool doFill = true, doEdge = false;
    if (kind == 0) {  // primary
        fill = disabled ? RGB(44, 48, 58) : pressed ? kBrandDown : hot ? kBrandHot : kBrand;
        ink = disabled ? kDim : RGB(255, 255, 255);
    } else if (kind == 1) {  // secondary
        fill = pressed ? RGB(28, 31, 38) : hot ? RGB(44, 49, 60) : RGB(34, 38, 46); edge = kFieldEdge; doEdge = true;
        ink = disabled ? kDim : kText;
    } else {  // quiet
        doFill = hot || pressed; fill = pressed ? RGB(26, 29, 35) : RGB(34, 38, 46);
        ink = disabled ? kDim : hot ? kText : kMuted;
    }
    {
        Gdiplus::Graphics gr(mem); gr.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
        const float r = SF(8);
        if (doFill) FillRound(gr, 0.5f, 0.5f, (float)w - 1, (float)h - 1, r, GC(fill));
        if (doEdge) StrokeRound(gr, 0, 0, (float)w, (float)h, r, GC(edge));
        if ((d->itemState & ODS_FOCUS) && !(d->itemState & ODS_NOFOCUSRECT)) StrokeRound(gr, 0, 0, (float)w, (float)h, r, GC(kind == 0 ? kLogoWhite : kBrand, 170), SF(1.5f));
    }
    wchar_t text[128]; GetWindowTextW(d->hwndItem, text, 128);
    HGDIOBJ oldFont = SelectObject(mem, kind == 0 ? fBold : fUi);
    SetBkMode(mem, TRANSPARENT); SetTextColor(mem, ink);
    DrawTextW(mem, text, -1, &all, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
    SelectObject(mem, oldFont);
    BitBlt(d->hDC, d->rcItem.left, d->rcItem.top, w, h, mem, 0, 0, SRCCOPY);
    SelectObject(mem, oldBmp); DeleteObject(bmp); DeleteDC(mem);
}

// Check box: BM_GETCHECK / BM_SETCHECK like a BS_AUTOCHECKBOX; BN_CLICKED to the parent after each toggle.
LRESULT CALLBACK CheckProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    auto checked = [h] { return GetPropW(h, L"wgCheck") != nullptr; };
    auto toggle = [h, &checked] {
        SetPropW(h, L"wgCheck", GetPropW(h, L"wgRadio") ? (HANDLE)1 : checked() ? nullptr : (HANDLE)1);
        InvalidateRect(h, nullptr, FALSE);
        SendMessageW(GetParent(h), WM_COMMAND, MAKEWPARAM(GetDlgCtrlID(h), BN_CLICKED), (LPARAM)h);
    };
    switch (m) {
        case BM_GETCHECK: return checked() ? BST_CHECKED : BST_UNCHECKED;
        case BM_SETCHECK: SetPropW(h, L"wgCheck", wp == BST_CHECKED ? (HANDLE)1 : nullptr); InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_SETFONT: SetPropW(h, L"wgFont", (HANDLE)wp); if (lp) InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_SETTEXT: { LRESULT r = DefWindowProcW(h, m, wp, lp); InvalidateRect(h, nullptr, FALSE); return r; }
        case WM_LBUTTONDOWN: SetFocus(h); SetCapture(h); return 0;
        case WM_LBUTTONUP:
            if (GetCapture() == h) {
                ReleaseCapture();
                POINT pt{(short)LOWORD(lp), (short)HIWORD(lp)}; RECT rc; GetClientRect(h, &rc);
                if (PtInRect(&rc, pt) && IsWindowEnabled(h)) toggle();
            }
            return 0;
        case WM_KEYUP: if (wp == VK_SPACE && IsWindowEnabled(h)) toggle(); return 0;
        case BM_CLICK: if (IsWindowEnabled(h)) toggle(); return 0;
        case WM_GETDLGCODE: return DLGC_BUTTON;
        case WM_SETFOCUS: case WM_KILLFOCUS: case WM_ENABLE: InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_SETCURSOR: SetCursor(LoadCursorW(nullptr, IDC_HAND)); return TRUE;
        case WM_ERASEBKGND: return 1;
        case WM_PAINT: case WM_PRINTCLIENT: {  // WM_PRINTCLIENT: drawn into the caller's DC (the launcher's hidden ui-dump)
            Kit& kit = *KitOf(h);
            PAINTSTRUCT ps{}; HDC dc = m == WM_PRINTCLIENT ? (HDC)wp : BeginPaint(h, &ps);
            RECT rc; GetClientRect(h, &rc);
            HDC mem = CreateCompatibleDC(dc); HBITMAP bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom); HGDIOBJ oldBmp = SelectObject(mem, bmp);
            FillRect(mem, &rc, kit.Brush(kit.BgOf(h)));
            const float box = kit.SF(18), top = kit.SF(2);
            {
                Gdiplus::Graphics gr(mem); gr.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
                if (GetPropW(h, L"wgRadio")) {
                    Gdiplus::SolidBrush outer(GC(kField));
                    Gdiplus::Pen edge(GC(checked() ? kBrand : kFieldEdge), kit.SF(1.5f));
                    gr.FillEllipse(&outer, 0.5f, top, box, box);
                    gr.DrawEllipse(&edge, 0.5f, top, box, box);
                    if (checked()) {
                        Gdiplus::SolidBrush dot(GC(IsWindowEnabled(h) ? kBrand : kDim));
                        gr.FillEllipse(&dot, kit.SF(5), top + kit.SF(4.5f), kit.SF(9), kit.SF(9));
                    }
                } else if (checked()) {
                    FillRound(gr, 0.5f, top, box, box, kit.SF(5), GC(IsWindowEnabled(h) ? kBrand : kDim));
                    CheckMark(gr, 0.5f, top, box, GC(RGB(255, 255, 255)), kit.SF(2.2f));
                } else {
                    FillRound(gr, 0.5f, top, box, box, kit.SF(5), GC(kField));
                    StrokeRound(gr, 0.5f, top, box, box, kit.SF(5), GC(kFieldEdge), kit.SF(1.5f));
                }
                if (GetFocus() == h) StrokeRound(gr, 0, top - 0.5f, box + 1, box + 1, kit.SF(5), GC(kBrand, 170), kit.SF(1.5f));
            }
            wchar_t text[1024]; GetWindowTextW(h, text, 1024);
            RECT tr{kit.S(30), 0, rc.right, rc.bottom};
            HGDIOBJ oldFont = SelectObject(mem, (HFONT)GetPropW(h, L"wgFont") ? (HFONT)GetPropW(h, L"wgFont") : kit.fUi);
            SetBkMode(mem, TRANSPARENT); SetTextColor(mem, IsWindowEnabled(h) ? kText : kDim);
            DrawTextW(mem, text, -1, &tr, DT_LEFT | DT_TOP | DT_WORDBREAK | DT_NOPREFIX);
            SelectObject(mem, oldFont);
            BitBlt(dc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
            SelectObject(mem, oldBmp); DeleteObject(bmp); DeleteDC(mem);
            if (m == WM_PAINT) EndPaint(h, &ps);
            return 0;
        }
        case WM_DESTROY: RemovePropW(h, L"wgCheck"); RemovePropW(h, L"wgFont"); RemovePropW(h, L"wgRadio"); break;
    }
    return DefWindowProcW(h, m, wp, lp);
}

// Progress bar: PBM_SETRANGE32 / PBM_SETPOS; a rounded track with the accent fill.
LRESULT CALLBACK ProgressProc(HWND h, UINT m, WPARAM wp, LPARAM lp) {
    switch (m) {
        case PBM_SETRANGE32: SetPropW(h, L"wgHi", (HANDLE)(INT_PTR)std::max<LPARAM>(1, lp)); InvalidateRect(h, nullptr, FALSE); return 0;
        case PBM_SETPOS: SetPropW(h, L"wgPos", (HANDLE)(INT_PTR)wp); InvalidateRect(h, nullptr, FALSE); return 0;
        case WM_ERASEBKGND: return 1;
        case WM_PAINT: case WM_PRINTCLIENT: {  // WM_PRINTCLIENT: drawn into the caller's DC (the launcher's hidden ui-dump)
            Kit& kit = *KitOf(h);
            PAINTSTRUCT ps{}; HDC dc = m == WM_PRINTCLIENT ? (HDC)wp : BeginPaint(h, &ps);
            RECT rc; GetClientRect(h, &rc);
            HDC mem = CreateCompatibleDC(dc); HBITMAP bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom); HGDIOBJ oldBmp = SelectObject(mem, bmp);
            FillRect(mem, &rc, kit.Brush(kit.BgOf(h)));
            const INT_PTR hi = (INT_PTR)GetPropW(h, L"wgHi"), pos = (INT_PTR)GetPropW(h, L"wgPos");
            const float frac = hi > 0 ? std::min(1.0f, std::max(0.0f, (float)pos / (float)hi)) : 0.0f;
            {
                Gdiplus::Graphics gr(mem); gr.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
                const float H = (float)rc.bottom, W = (float)rc.right;
                FillRound(gr, 0, 0, W, H, H / 2, GC(RGB(36, 40, 49)));
                if (frac > 0) {
                    Gdiplus::LinearGradientBrush lg(Gdiplus::PointF(0, 0), Gdiplus::PointF(W, 0), GC(kBrandDown), GC(kBrandHot));
                    Gdiplus::GraphicsPath p; RoundPath(p, 0, 0, std::max(H, W * frac), H, H / 2);
                    gr.FillPath(&lg, &p);
                }
            }
            BitBlt(dc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
            SelectObject(mem, oldBmp); DeleteObject(bmp); DeleteDC(mem);
            EndPaint(h, &ps);
            return 0;
        }
        case WM_DESTROY: RemovePropW(h, L"wgHi"); RemovePropW(h, L"wgPos"); break;
    }
    return DefWindowProcW(h, m, wp, lp);
}

void Kit::RegisterClasses() {
    static bool registered = false;
    if (!registered) {
        WNDCLASSEXW wc{}; wc.cbSize = sizeof(wc); wc.hInstance = hinst; wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.lpfnWndProc = CheckProc; wc.lpszClassName = L"WrongGPUCheck"; RegisterClassExW(&wc);
        wc.lpfnWndProc = ProgressProc; wc.lpszClassName = L"WrongGPUProgress"; RegisterClassExW(&wc);
        registered = true;
    }
}
void Kit::Attach(HWND top) { SetPropW(top, L"wgKit", this); }

void Kit::CreateFonts(UINT dpi) {
    NONCLIENTMETRICSW ncm{}; ncm.cbSize = sizeof(ncm);
    SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof(ncm), &ncm, 0);
    auto font = [&](int pt, int weight, const wchar_t* face) {
        LOGFONTW lf = ncm.lfMessageFont; lf.lfHeight = -MulDiv(pt, dpi, 72); lf.lfWeight = weight; lf.lfQuality = CLEARTYPE_QUALITY;
        if (face) wcsncpy(lf.lfFaceName, face, LF_FACESIZE - 1);
        return CreateFontIndirectW(&lf);
    };
    auto has = [](const wchar_t* face) {
        LOGFONTW lf{}; lf.lfCharSet = DEFAULT_CHARSET; wcsncpy(lf.lfFaceName, face, LF_FACESIZE - 1);
        bool found = false;
        HDC dc = GetDC(nullptr);
        EnumFontFamiliesExW(dc, &lf, [](const LOGFONTW*, const TEXTMETRICW*, DWORD, LPARAM p) -> int { *(bool*)p = true; return 0; }, (LPARAM)&found, 0);
        ReleaseDC(nullptr, dc);
        return found;
    };
    // Windows 11's Segoe UI Variable when present, else Segoe UI; icons from Segoe Fluent Icons / Segoe MDL2 Assets.
    const wchar_t* textFace = has(L"Segoe UI Variable Text") ? L"Segoe UI Variable Text" : L"Segoe UI";
    const wchar_t* displayFace = has(L"Segoe UI Variable Display") ? L"Segoe UI Variable Display" : L"Segoe UI";
    const wchar_t* iconFace = has(L"Segoe Fluent Icons") ? L"Segoe Fluent Icons" : has(L"Segoe MDL2 Assets") ? L"Segoe MDL2 Assets" : nullptr;
    fUi = font(10, FW_NORMAL, textFace); fBold = font(10, FW_SEMIBOLD, textFace); fTitle = font(20, FW_SEMIBOLD, displayFace);
    fSub = font(10, FW_NORMAL, textFace); fMono = font(9, FW_NORMAL, L"Consolas"); fSmall = font(9, FW_NORMAL, textFace);
    fCardTitle = font(12, FW_SEMIBOLD, displayFace); fLogo = font(17, FW_BOLD, displayFace);
    iconFont = iconFace != nullptr;
    if (iconFace) fIcon = font(14, FW_NORMAL, iconFace);
    white = CreateSolidBrush(RGB(255, 255, 255));
}

void Kit::LoadLogo(const void* png, DWORD size) {
    HGLOBAL mem = png && size ? GlobalAlloc(GMEM_MOVEABLE, size) : nullptr;
    if (mem) {
        std::memcpy(GlobalLock(mem), png, size);
        GlobalUnlock(mem);
        IStream* stream = nullptr;
        if (SUCCEEDED(CreateStreamOnHGlobal(mem, TRUE, &stream))) {
            logo = Gdiplus::Bitmap::FromStream(stream);
            if (logo && logo->GetLastStatus() != Gdiplus::Ok) { delete logo; logo = nullptr; }
            stream->Release();
        } else {
            GlobalFree(mem);
        }
    }
}

void SaveWindowBitmap(HWND hwnd, const wstring& path) {
    RECT rc; GetWindowRect(hwnd, &rc);
    int w = rc.right - rc.left, h = rc.bottom - rc.top;
    HDC sdc = GetDC(nullptr), mdc = CreateCompatibleDC(sdc);
    HBITMAP bmp = CreateCompatibleBitmap(sdc, w, h);
    HGDIOBJ old = SelectObject(mdc, bmp);
    PrintWindow(hwnd, mdc, PW_RENDERFULLCONTENT);
    BITMAPINFOHEADER bi{}; bi.biSize = sizeof(bi); bi.biWidth = w; bi.biHeight = -h; bi.biPlanes = 1; bi.biBitCount = 32; bi.biCompression = BI_RGB;
    std::vector<char> px((size_t)w * h * 4);
    GetDIBits(mdc, bmp, 0, h, px.data(), (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    if (std::all_of(px.begin(), px.end(), [](char c) { return c == 0; })) {  // PrintWindow unsupported (e.g. Wine): copy from the screen
        HDC wdc = GetWindowDC(hwnd);
        BitBlt(mdc, 0, 0, w, h, wdc, 0, 0, SRCCOPY);
        ReleaseDC(hwnd, wdc);
        GetDIBits(mdc, bmp, 0, h, px.data(), (BITMAPINFO*)&bi, DIB_RGB_COLORS);
    }
    BITMAPFILEHEADER fh{}; fh.bfType = 0x4D42; fh.bfSize = (DWORD)(sizeof(fh) + sizeof(bi) + px.size()); fh.bfOffBits = sizeof(fh) + sizeof(bi);
    HANDLE f = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f != INVALID_HANDLE_VALUE) { DWORD wr; WriteFile(f, &fh, sizeof(fh), &wr, nullptr); WriteFile(f, &bi, sizeof(bi), &wr, nullptr); WriteFile(f, px.data(), (DWORD)px.size(), &wr, nullptr); CloseHandle(f); }
    SelectObject(mdc, old); DeleteObject(bmp); DeleteDC(mdc); ReleaseDC(nullptr, sdc);
}

void DarkTitleBar(HWND hwnd, COLORREF caption, COLORREF border) {
    BOOL dark = TRUE;
    DwmSetWindowAttribute(hwnd, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &dark, sizeof dark);
    DwmSetWindowAttribute(hwnd, 35 /* DWMWA_CAPTION_COLOR */, &caption, sizeof caption);
    DwmSetWindowAttribute(hwnd, 34 /* DWMWA_BORDER_COLOR */, &border, sizeof border);
}

}  // namespace wg::ui
