// SPDX-License-Identifier: MIT
// WrongGPU launcher window: the colours that carry meaning, without any drawing (ui_shell.h's Theme is made from them; the
// launcher's unit tests check their contrast). The "Unlock palette" (build/launcher/ui-options/BE-wide/notes.md, owner
// 2026-11-01): green means WrongGPU is working (status dots, installed / worked / checks passed, the end of a progress
// run) and is fixed; red means something needs you; the dark theme's page is true black. The accent colours the controls
// (Play, switches on, links, Back, the chosen radio, Continue): Green by default, or Blue, Purple, Pink, Cyan (the old
// Signal tint), Graphite, or the Windows accent colour. Red and amber are never an accent. Written for this project.
#pragma once
#include <algorithm>
#include <cmath>
#include <string>

namespace wg::app::lui {

struct Col { float r = 0, g = 0, b = 0, a = 1; };
constexpr Col Hex(unsigned rgb, float a = 1.f) { return Col{((rgb >> 16) & 255) / 255.f, ((rgb >> 8) & 255) / 255.f, (rgb & 255) / 255.f, a}; }
inline Col Alpha(Col c, float a) { c.a *= a; return c; }
inline Col Mix(Col a, Col b, float t) { return Col{a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t}; }

// WCAG 2 relative luminance and contrast ratio (alpha ignored: opaque colours)
inline float Luminance(Col c) {
    auto lin = [](float v) { return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f); };
    return 0.2126f * lin(c.r) + 0.7152f * lin(c.g) + 0.0722f * lin(c.b);
}
inline float Contrast(Col a, Col b) {
    const float la = Luminance(a), lb = Luminance(b);
    return (std::max(la, lb) + 0.05f) / (std::min(la, lb) + 0.05f);
}

// The surfaces the accent's colours are measured on: the page, a grouped list's cell, the fill (search, wells)
struct Surfaces { Col bg, cell, fill; };
inline Surfaces SurfacesOf(bool dark) {
    if (dark) return Surfaces{Hex(0x000000), Hex(0x161618), Hex(0x232326)};
    return Surfaces{Hex(0xf3f3f5), Hex(0xffffff), Hex(0xececf0)};
}

// The fixed "working" green (never follows the accent): graphics (dots, the progress run's end) and words
inline Col StatusGreen(bool dark) { return dark ? Hex(0x4bd865) : Hex(0x1f8f3e); }
inline Col StatusGreenText(bool dark) { return dark ? Hex(0x6fe38a) : Hex(0x18733a); }

// The accent picker's choices, in the order Settings shows them (launcher.ini "accent=" keeps the key)
enum Accent { A_GREEN, A_BLUE, A_PURPLE, A_PINK, A_CYAN, A_GRAPHITE, A_WINDOWS, A_COUNT };
inline const wchar_t* AccentKey(int a) {
    static const wchar_t* const k[A_COUNT] = {L"green", L"blue", L"purple", L"pink", L"cyan", L"graphite", L"windows"};
    return k[a >= 0 && a < A_COUNT ? a : 0];
}
inline const wchar_t* AccentName(int a) {
    static const wchar_t* const k[A_COUNT] = {L"Green", L"Blue", L"Purple", L"Pink", L"Cyan", L"Graphite", L"Windows"};
    return k[a >= 0 && a < A_COUNT ? a : 0];
}
inline int AccentFromKey(const std::wstring& key) {
    for (int a = 0; a < A_COUNT; ++a) if (key == AccentKey(a)) return a;
    return A_GREEN;
}

// One accent in one theme: graphics (switch track, the chosen radio, the swatch's ring: 3:1 on the page and cells), a fill
// under white words (Play, Continue: 4.5:1), words (links, Back: 4.5:1 on the page, cells, the fill and the tint), the
// pills' tint and its hover.
struct AccentTokens { Col c, f, t, tint, tintHover; };

// The Windows accent colour made into tokens with the same rules as the fixed accents: the fill darkened until white words
// read at 4.6:1, the graphics and the words moved away from the surfaces until they reach 3.2:1 and 4.6:1, the tint a
// light (or dark) wash of the colour.
inline AccentTokens DeriveAccent(Col base, bool dark) {
    const Surfaces s = SurfacesOf(dark);
    const Col white = Hex(0xffffff), black = Hex(0x000000);
    auto toward = [](Col c, Col to, const auto& ok) { for (int i = 0; i < 60 && !ok(c); ++i) c = Mix(c, to, 0.05f); return c; };
    AccentTokens k;
    base.a = 1;
    k.f = toward(base, black, [&](Col c) { return Contrast(white, c) >= 4.6f; });
    if (dark) {
        k.c = toward(base, white, [&](Col c) { return Contrast(c, s.cell) >= 3.2f && Contrast(c, s.fill) >= 3.0f; });
        k.tint = Mix(Hex(0x0e0e10), base, 0.17f);
        k.tintHover = Mix(Hex(0x0e0e10), base, 0.23f);
        k.t = toward(Mix(base, white, 0.3f), white, [&](Col c) {
            return Contrast(c, s.cell) >= 4.6f && Contrast(c, s.fill) >= 4.6f && Contrast(c, k.tintHover) >= 4.6f;
        });
    } else {
        k.c = toward(base, black, [&](Col c) { return Contrast(c, s.bg) >= 3.2f; });
        k.tint = Mix(white, base, 0.11f);
        k.tintHover = Mix(white, base, 0.16f);
        k.t = toward(k.f, black, [&](Col c) {
            return Contrast(c, s.bg) >= 4.6f && Contrast(c, s.fill) >= 4.6f && Contrast(c, k.tintHover) >= 4.6f;
        });
    }
    return k;
}

// The accent's tokens (the mockup's ACCENTS table, BE-wide/index.html; Green is its "signal" row under the Unlock palette,
// Cyan its old Signal tint). windowsRgb: the Windows accent colour (0xRRGGBB) for A_WINDOWS.
inline AccentTokens AccentFor(int accent, bool dark, unsigned windowsRgb) {
    struct Row { unsigned c, f, t, ti, th; };
    static const Row kDark[A_WINDOWS] = {
        {0x4bd865, 0x1a7f35, 0x6fe38a, 0x10261a, 0x16321f},  // green
        {0x3d8bfd, 0x1f68d6, 0x8ab8ff, 0x1b2a40, 0x22334d},  // blue
        {0xa07cf0, 0x7046cf, 0xc3a6ff, 0x2a2240, 0x332a4d},  // purple
        {0xff6aa8, 0xb8306f, 0xff9cc8, 0x3a2030, 0x47263a},  // pink
        {0x2bb7cf, 0x0f7a8d, 0x6fd6e8, 0x173238, 0x1d3e45},  // cyan
        {0xa4a4ad, 0x5a5a63, 0xc9c9d1, 0x2c2c31, 0x36363c},  // graphite
    };
    static const Row kLight[A_WINDOWS] = {
        {0x1f8f3e, 0x1a7f35, 0x18733a, 0xe3f4e7, 0xd4eedb},
        {0x1f68d6, 0x1f68d6, 0x1a5bbf, 0xe6effd, 0xd8e6fb},
        {0x7046cf, 0x7046cf, 0x6439c2, 0xefe9fc, 0xe5dcfa},
        {0xb8306f, 0xb8306f, 0xa82964, 0xfbe7f0, 0xf7d9e7},
        {0x00839c, 0x0f7a8d, 0x006f85, 0xe2f4f7, 0xd2edf2},
        {0x5a5a63, 0x5a5a63, 0x4e4e56, 0xececef, 0xe2e2e6},
    };
    if (accent == A_WINDOWS) return DeriveAccent(Hex(windowsRgb & 0xffffff), dark);
    const Row& r = (dark ? kDark : kLight)[accent >= 0 && accent < A_WINDOWS ? accent : 0];
    return AccentTokens{Hex(r.c), Hex(r.f), Hex(r.t), Hex(r.ti), Hex(r.th)};
}

// A filled button that is off (Play, Continue, Update N games, Apply, a banner's Update or "Repairing..."; red or the
// accent, either theme): one rule for all of them (SHIP-BAR 2: disabled states distinct and consistent). The fill fades
// toward the cell and the label reads at 3.5:1 on it: readable (disabled text 3:1 or better), never as strong as a live
// label (4.5:1 or better), so an off button never reads as a live darker one. Dark: the fill faded 55 % toward the cell,
// the label that fill mixed toward white until 3.5:1; light: the label white, the fill faded until white is down to 3.5:1.
struct FilledOff { Col fill, label; };
constexpr float kOffLabelContrast = 3.5f;
inline FilledOff DisabledFilled(Col base, Col cell, Col on, bool dark) {
    FilledOff o;
    base.a = 1;
    if (dark) {
        o.fill = Mix(base, cell, 0.55f);
        o.label = o.fill;
        for (int i = 1; i <= 100 && Contrast(o.label, o.fill) < kOffLabelContrast; ++i) o.label = Mix(o.fill, on, i / 100.f);
    } else {
        o.fill = base;
        o.label = on;
        for (int i = 1; i <= 100; ++i) {
            const Col f = Mix(base, cell, i / 100.f);
            if (Contrast(on, f) < kOffLabelContrast) break;
            o.fill = f;
        }
    }
    return o;
}

// The swatch Settings draws for an accent (the mockup's swatch colours, the same in both themes)
inline Col AccentSwatch(int accent, unsigned windowsRgb) {
    static const unsigned kSw[A_WINDOWS] = {0x4bd865, 0x3d8bfd, 0xa07cf0,  // green, blue, purple
                                            0xff6aa8, 0x2bb7cf, 0x9a9aa3}; // pink, cyan, graphite
    return accent == A_WINDOWS ? Hex(windowsRgb & 0xffffff) : Hex(kSw[accent >= 0 && accent < A_WINDOWS ? accent : 0]);
}

}  // namespace wg::app::lui
