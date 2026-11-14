// SPDX-License-Identifier: MIT
// Linux: draws a fixed sample through Gfx (Cairo and Pango, src/app/ui_gfx_cairo.cpp) offscreen at scale 1 and 2, writes the
// PNGs to the folder given as argument, and checks what a picture alone would not tell a test: the measurements, the dry pass,
// the fonts being the vendored ones, a fade, a clip, an unmapped icon. Looking at the PNGs is still the real check of the look.
#include "app/ui_shell.h"

#include <sys/stat.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace wg::app::lui {
Theme pal;  // the real one is ui_widgets.cpp's (Windows); the drawing here needs only the colours it reads
}
using namespace wg::app::lui;
using std::wstring;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static ImageP GradientImage(int w, int h) {
    auto im = std::make_shared<Image>();
    im->w = w; im->h = h;
    im->px.resize((size_t)w * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t* p = &im->px[((size_t)y * w + x) * 4];
            p[0] = (uint8_t)(255 * x / (w - 1)); p[1] = (uint8_t)(255 * y / (h - 1)); p[2] = 90; p[3] = 255;  // BGRA, opaque
        }
    return im;
}

static bool Same(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); }

// The sample: every call the pages make a lot of, on a 480 x 360 DIP page
static void Draw(Gfx& gx, const ImageP& img) {
    gx.Clear(Hex(0xF3F3F3));
    // cards with a shadow, per-corner radii, a stroke
    gx.Shadow(Rc{20, 20, 200, 90}, 10, 16, 0.18f, 4);
    gx.FillRound(Rc{20, 20, 200, 90}, 10, Hex(0xFFFFFF));
    gx.StrokeRound(Rc{20, 20, 200, 90}, 10, Hex(0x000000, 0.12f), 1);
    gx.FillRound4(Rc{240, 20, 100, 90}, 20, 4, 20, 4, Hex(0x2D7D46));
    gx.Grad(Rc{350, 20, 110, 90}, 12, Hex(0x3B82F6), Hex(0xA855F7), 0, 0, 1, 1);
    gx.GradStops(Rc{20, 124, 200, 14}, 7, {{0.f, Hex(0xEF4444)}, {0.5f, Hex(0xF59E0B)}, {1.f, Hex(0x22C55E)}}, 0, 0, 1, 0, 0.8f);
    gx.Line(240, 131, 340, 131, Hex(0x111111), 2);
    gx.Circle(364, 131, 6, Hex(0x2D7D46));
    gx.Ring(388, 131, 6, Hex(0x2D7D46), 2);
    gx.Glow(440, 131, 26, 12, Hex(0x3B82F6, 0.8f));
    // text: each face and weight, wrapped, trimmed, centred
    TS t;
    t.color = Hex(0x111111);
    t.size = 13; gx.Text(L"Regular text, Selawik 13", Rc{20, 150, 220, 18}, t);
    t.weight = 600; gx.Text(L"Semibold text", Rc{20, 170, 220, 18}, t);
    t.weight = 700; gx.Text(L"Bold text", Rc{20, 190, 220, 18}, t);
    t.weight = 300; gx.Text(L"Light text", Rc{20, 210, 220, 18}, t);
    t.weight = 400; t.face = F_MONO; gx.Text(L"mono 0123456789", Rc{20, 230, 220, 18}, t);
    t.face = F_DISPLAY; t.size = 22; t.weight = 600; gx.Text(L"Display 22", Rc{250, 146, 220, 30}, t);
    t = TS{}; t.color = Hex(0x111111); t.wrap = true;
    gx.Text(L"Wrapped text breaks at the box's width and carries on to as many lines as it needs.", Rc{250, 180, 210, 60}, t);
    t.trim = true; t.color = Hex(0x444444);
    gx.Text(L"Trimmed wrapped text keeps the lines that fit the box and ends the last one with an ellipsis sign.", Rc{250, 250, 210, 34}, t);
    t = TS{}; t.color = Hex(0x111111);
    gx.Text(L"One line is cut with an ellipsis when it is too long for its box", Rc{20, 252, 200, 18}, t);
    t.align = 1; gx.Text(L"Centred", Rc{20, 272, 200, 18}, t);
    t.align = 2; t.underline = true; gx.Text(L"Right, underlined", Rc{20, 292, 200, 18}, t);
    t = TS{}; t.color = Hex(0x555555); t.spacing = 2; t.lineH = 20; t.wrap = true;
    gx.Text(L"Spaced and tall lines of text", Rc{20, 312, 140, 40}, t);
    t = TS{}; t.color = Hex(0x111111); t.wrap = true;
    gx.TextLead(L"Ready. The files are in place and nothing else needs doing.", 6, Hex(0x2D7D46), 700, Rc{170, 312, 290, 40}, t);
    // icons
    const wchar_t icons[] = {ic::check, ic::warn, ic::gear, ic::search, ic::folder, ic::download, ic::trash, ic::chevron};
    for (int i = 0; i < 8; ++i) gx.Icon(icons[i], 30 + i * 26.f, 350.f, 16, Hex(0x111111));
    // an image: cover in a rounded box, contain with a shadow, nearest
    gx.ImageCover(img, Rc{20, 360, 100, 60}, 8, 0.5f);
    gx.ImageContain(img, Rc{130, 360, 90, 60}, 1, 1, 1, 10);
    gx.ImageNearest(img, Rc{230, 360, 60, 60}, 6);
    // a clip and a fade
    gx.PushClip(Rc{300, 360, 50, 50});
    gx.Fill(Rc{280, 340, 90, 90}, Hex(0x3B82F6));
    gx.PopClip();
    gx.PushRoundClip(Rc{360, 360, 50, 50}, 25);
    gx.Fill(Rc{350, 350, 100, 100}, Hex(0xA855F7));
    gx.PopRoundClip();
    gx.PushFade(0.5f);
    gx.FillRound(Rc{420, 360, 50, 50}, 8, Hex(0x000000));
    gx.PopFade();
}

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : "gfx-smoke";
    mkdir(out.c_str(), 0755);
    pal.black = Hex(0x000000);
    pal.artShade = Hex(0x000000);
    pal.text = Hex(0x111111);

    Gfx gx;
    CHECK(gx.Init());
    CHECK(gx.iconFont);  // the vendored Fluent System Icons were found and registered
    const ImageP img = GradientImage(64, 40);
    const int W = 480, H = 430;

    // measurements (no target needed)
    TS t;
    t.size = 13;
    CHECK(gx.TextW(L"Hello", t) > 0);
    CHECK(gx.TextW(L"Hello world", t) > gx.TextW(L"Hello", t));
    CHECK(gx.TextW(L"Hello", TS{}) > 0 && gx.TextW(L"W", t) > gx.TextW(L"i", t));
    TS big = t; big.size = 26;
    CHECK(gx.TextW(L"Hello", big) > 1.8f * gx.TextW(L"Hello", t));
    TS bold = t; bold.weight = 700;
    CHECK(gx.TextW(L"Hello world", bold) > gx.TextW(L"Hello world", t));  // Selawik Bold is really used, not a fallback
    TS wrap = t; wrap.wrap = true;
    const wstring para = L"A paragraph of words that must wrap when the box is narrow but fits on one line when it is wide.";
    const float wide = gx.TextH(para, 2000, wrap), narrow = gx.TextH(para, 120, wrap);
    CHECK(wide > 0 && narrow > 1.9f * wide);
    TS tall = wrap; tall.lineH = 30;
    CHECK(gx.TextH(para, 120, tall) > narrow);
    float ex = 0, ey = 0, ex2 = 0, ey2 = 0;
    gx.TextEnd(L"abc", 200, t, ex, ey);
    gx.TextEnd(L"abcdef", 200, t, ex2, ey2);
    CHECK(ex > 0 && ex2 > ex);
    gx.TextEnd(para, 120, wrap, ex, ey);
    CHECK(ey > 0);  // the last character is on a later line

    for (float s : {1.f, 2.f}) {
        CHECK(gx.BeginOffscreen(W, H, s));
        Draw(gx, img);
        gx.EndFrame();
        std::vector<uint8_t> before;
        UINT pw = 0, ph = 0;
        CHECK(gx.Pixels(before, pw, ph));
        CHECK(pw == (UINT)(W * s) && ph == (UINT)(H * s));
        char name[64];
        std::snprintf(name, sizeof name, "/smoke-%dx.png", (int)s);
        const wstring path = wstring(out.begin(), out.end()) + wstring(name, name + std::strlen(name));
        CHECK(gx.SavePng(path));
        struct stat st{};
        CHECK(stat((out + name).c_str(), &st) == 0 && st.st_size > 5000);

        auto at = [&](float x, float y) { const uint8_t* p = &before[((size_t)(y * s) * pw + (size_t)(x * s)) * 4]; return std::array<int, 4>{p[0], p[1], p[2], p[3]}; };
        CHECK(at(2, 2)[0] == 0xF3 && at(2, 2)[3] == 255);            // the cleared page
        CHECK(at(120, 100)[0] == 255 && at(120, 100)[1] == 255);     // the card's white
        CHECK(at(120, 114)[0] < 0xF3 && at(120, 114)[0] > 0xC0);    // the card's shadow darkens the page under it, softly
        CHECK(at(120, 122)[0] > at(120, 114)[0]);                    // and fades with the distance
        CHECK(at(290, 60)[1] > at(290, 60)[0]);                      // the green card (BGRA: G over B)
        CHECK(at(325, 385)[0] > 200 && at(325, 385)[2] < 120);       // inside the clip: blue (B high, R low)
        CHECK(at(290, 385)[0] == 0xF3);                              // the clip kept the blue off the page outside it
        CHECK(at(445, 385)[2] > 100 && at(445, 385)[2] < 150);       // black at half opacity over 0xF3: about 121
        CHECK(at(364, 131)[1] > 100 && at(364, 131)[2] < 100);       // the circle

        // the dry pass draws nothing and an unmapped icon draws nothing
        gx.dry = true;
        Draw(gx, img);
        gx.dry = false;
        std::vector<uint8_t> after;
        CHECK(gx.Pixels(after, pw, ph) && Same(before, after));
        gx.Icon(0xE000, 30, 30, 16, Hex(0xFF0000));
        CHECK(gx.Pixels(after, pw, ph) && Same(before, after));
    }

    // a rotation stays open through a clip, a round clip and a fade that are popped inside it (the matrix is not rebuilt from identity)
    for (int kind = 0; kind < 3; ++kind) {
        CHECK(gx.BeginOffscreen(100, 100, 1));
        gx.Clear(Hex(0xFFFFFF));
        gx.PushRotate(50, 50, 90);  // clockwise: a bar to the right of the centre turns to below it
        if (kind == 0) { gx.PushClip(Rc{0, 0, 100, 100}); gx.PopClip(); }
        else if (kind == 1) { gx.PushRoundClip(Rc{0, 0, 100, 100}, 4); gx.PopRoundClip(); }
        else { gx.PushFade(1.f); gx.PopFade(); }
        gx.Fill(Rc{70, 45, 20, 10}, Hex(0x000000));
        gx.PopRotate();
        std::vector<uint8_t> rp;
        UINT rw = 0, rh = 0;
        CHECK(gx.Pixels(rp, rw, rh));
        auto dark = [&](int x, int y) { return rp[((size_t)y * rw + x) * 4] < 64; };
        CHECK(dark(50, 80));    // the bar at the bottom: the rotation survived the pop
        CHECK(!dark(80, 50));   // and not where it would be unrotated
        gx.EndFrame();
    }

    // text and icons really leave ink: a blank page, then one glyph
    CHECK(gx.BeginOffscreen(60, 40, 2));
    gx.Clear(Hex(0xFFFFFF));
    gx.Icon(ic::check, 30, 20, 20, Hex(0x000000));
    std::vector<uint8_t> px;
    UINT w = 0, h = 0;
    CHECK(gx.Pixels(px, w, h));
    int ink = 0;
    for (size_t i = 0; i < px.size(); i += 4) if (px[i] < 200) ++ink;
    CHECK(ink > 20);
    gx.EndFrame();

    // a backdrop blur softens a hard edge, keeps the page outside the rounded shape, and a fade/clip open makes it decline
    CHECK(gx.BeginOffscreen(100, 60, 1));
    gx.Fill(Rc{0, 0, 50, 60}, Hex(0x000000));
    gx.Fill(Rc{50, 0, 50, 60}, Hex(0xFFFFFF));
    CHECK(gx.Backdrop(Rc{20, 10, 60, 40}, 8, 4, 1));
    CHECK(gx.Pixels(px, w, h));
    const int mid = px[((size_t)30 * w + 50) * 4], edge = px[((size_t)30 * w + 30) * 4], outside = px[((size_t)30 * w + 10) * 4];
    CHECK(mid > 60 && mid < 200 && edge < mid && outside == 0);
    gx.PushFade(0.5f);
    CHECK(!gx.Backdrop(Rc{20, 10, 60, 40}, 8, 4, 1));
    gx.PopFade();
    gx.Freeze(7);
    gx.Fill(Rc{0, 0, 100, 60}, Hex(0xFF0000));
    CHECK(gx.DrawFrozen(7) && !gx.DrawFrozen(8));
    CHECK(gx.Pixels(px, w, h) && px[((size_t)30 * w + 10) * 4 + 2] == 0);  // the frozen picture is back: no red
    gx.EndFrame();

    if (fails) { std::fprintf(stderr, "gfx-linux-smoke: %d failure(s)\n", fails); return 1; }
    std::printf("gfx-linux-smoke: ok (PNGs in %s)\n", out.c_str());
    return 0;
}
