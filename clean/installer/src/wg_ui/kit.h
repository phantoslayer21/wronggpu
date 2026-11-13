// SPDX-License-Identifier: MIT
// WrongGPU UI kit (wg_ui): the owner-drawn widgets of the installer's wizard - buttons, check boxes and radio buttons,
// a progress bar, cards, the palette of WrongGPU's in-game overlay, fonts, the wordmark, the dark title bar - for the
// wizard and the launcher. Windows' own GDI, GDI+, DWM and comctl32 only (no third-party code). Moved from setup.cpp.
// Written for this project.
#pragma once
#include <map>
#include <string>

#include "wg_core/base.h"
namespace Gdiplus {  // gdiplus.h needs min / max (NOMINMAX is defined)
using std::max;
using std::min;
}  // namespace Gdiplus
#include <gdiplus.h>

namespace wg::ui {

// Palette (the overlay's): backgrounds, text, accents.
const COLORREF kBg = RGB(20, 22, 27), kSide = RGB(14, 15, 17), kCard = RGB(27, 30, 37), kCardEdge = RGB(40, 44, 53), kField = RGB(14, 16, 20),
               kFieldEdge = RGB(54, 59, 70), kText = RGB(236, 238, 242), kMuted = RGB(150, 156, 168), kDim = RGB(96, 102, 114),
               kGreen = RGB(77, 219, 120), kRed = RGB(255, 107, 107), kAmber = RGB(255, 189, 61), kGray = RGB(138, 143, 152), kInk = kText,
               // the logo's colours: brand red (accent: buttons, steps, progress, focus) and its off-white
               kBrand = RGB(253, 70, 60), kBrandHot = RGB(255, 104, 94), kBrandDown = RGB(212, 50, 42), kLogoWhite = RGB(242, 240, 237);

// The state of the widgets of one top-level window: DPI scale, fonts, brushes, the background each control sits on,
// the kind of each button, the button under the mouse, the wordmark. The check box and progress bar classes find it
// through the top-level window (Attach); the buttons through their subclass.
struct Kit {
    HINSTANCE hinst{};
    double scale = 1.0;
    HFONT fUi{}, fBold{}, fTitle{}, fSub{}, fMono{}, fSmall{}, fCardTitle{}, fLogo{}, fIcon{};
    HBRUSH white{};
    std::map<HWND, COLORREF> bg;        // background a control sits on (cards, fields)
    std::map<HWND, int> buttonKind;     // 0 primary, 1 secondary, 2 quiet
    std::map<COLORREF, HBRUSH> brushes;
    HWND hover{};                       // button under the mouse
    bool iconFont = false;              // Segoe Fluent Icons / Segoe MDL2 Assets present
    Gdiplus::Bitmap* logo = nullptr;    // the wordmark from the exe's resources (kLogoResourceId), or null

    int S(int v) const;      // logical pixels (96 DPI) to pixels
    float SF(float v) const;
    HBRUSH Brush(COLORREF c);
    COLORREF BgOf(HWND h) const;
    void SetBg(HWND h, COLORREF c);
    // A card: rounded panel with a hairline edge (logical coordinates).
    void Card(Gdiplus::Graphics& gr, int x, int y, int w, int h, COLORREF fill = kCard, COLORREF edge = kCardEdge) const;
    // GDI text in a logical rectangle.
    void UiText(HDC dc, const std::wstring& t, int x, int y, int w, int h, HFONT f, COLORREF c, UINT fmt = DT_LEFT | DT_TOP | DT_WORDBREAK | DT_NOPREFIX) const;
    void DrawButton(const DRAWITEMSTRUCT* d);  // a BS_OWNERDRAW button (WM_DRAWITEM)
    void RegisterClasses();                    // "WrongGPUCheck" (check box / radio button), "WrongGPUProgress"
    void Attach(HWND top);                     // the check box and progress bar classes of `top` use this kit
    void CreateFonts(UINT dpi);                // Segoe UI Variable (else Segoe UI), Consolas, Segoe Fluent Icons
    void LoadLogo(const void* png, DWORD size);  // the wordmark PNG, decoded by GDI+ from a copy
};

Gdiplus::Color GC(COLORREF c, BYTE a = 255);
void RoundPath(Gdiplus::GraphicsPath& p, float x, float y, float w, float h, float r);
void FillRound(Gdiplus::Graphics& gr, float x, float y, float w, float h, float r, const Gdiplus::Color& c);
void StrokeRound(Gdiplus::Graphics& gr, float x, float y, float w, float h, float r, const Gdiplus::Color& c, float width = 1);
void CheckMark(Gdiplus::Graphics& gr, float x, float y, float s, const Gdiplus::Color& c, float width);
void CrossMark(Gdiplus::Graphics& gr, float x, float y, float s, const Gdiplus::Color& c, float width);
// Owner-drawn buttons: hover tracking and the hand cursor (SetWindowSubclass with the Kit as reference data).
LRESULT CALLBACK ButtonSubclass(HWND h, UINT m, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR kit);
LRESULT CALLBACK CheckProc(HWND h, UINT m, WPARAM wp, LPARAM lp);
LRESULT CALLBACK ProgressProc(HWND h, UINT m, WPARAM wp, LPARAM lp);
// The window as a 32-bit BMP (development screenshots: PrintWindow, else a copy from the screen).
void SaveWindowBitmap(HWND hwnd, const std::wstring& path);
// Dark title bar (Windows 10 2004+ / Windows 11; ignored elsewhere).
void DarkTitleBar(HWND hwnd, COLORREF caption, COLORREF border);

}  // namespace wg::ui
