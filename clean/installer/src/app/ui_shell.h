// SPDX-License-Identifier: MIT
// WrongGPU launcher window (option E "Quiet", build/launcher/ui-options/E-quiet/index.html, light and dark): what the window's source files share. The window is
// painted with Direct2D and DirectWrite (WIC for images, DWM for the title bar); widgets are drawn per frame and leave
// "hits" (rectangle, id, role, name, state) that the mouse, the keyboard, UI Automation (ui_uia.cpp) and the ui-dump
// JSON use. Native EDIT children only for typed text (search, release URL, OptiScaler values). OS components only, no
// third-party code. Written for this project.
//   ui_gfx.cpp      Direct2D / DirectWrite / WIC drawing, PNG output
//   ui_art.cpp      game art from the player's own Steam cache, decoded off the UI thread
//   ui_widgets.cpp  buttons, switches, segmented controls, grouped lists, covers, menus, dialogs, toasts, the toolbar
//   ui_library.cpp  the Library page        ui_game.cpp  a game's page        ui_settings.cpp  Settings, first run
//   ui_uia.cpp      UI Automation provider  launcher_ui.cpp  the window, input, operations, ui-dump / ui-render
#pragma once
#include "app/launcher_app.h"

#include <d2d1_1.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dwrite_1.h>
#include <wincodec.h>

#include <functional>
#include <memory>
#include <mutex>

#include "app/ui_palette.h"
#include "wg_core/optikeys.h"

namespace wg::app::lui {

// ---- colours: the theme tokens (option E's two token sets: the mockup's :root and html[data-theme="dark"] custom properties) ----
// Col, Hex, Alpha, Mix, the contrast ratio and the accent colours: ui_palette.h (no drawing: the unit tests use it too)
// Quiet: type and space instead of boxes and rules. A soft page (true black in the dark theme), inset grouped lists with
// hairlines. Every colour the window draws comes from here; SetTheme switches the whole set.
// Colour has a meaning (the "Unlock palette", build/launcher/ui-options/BE-wide/notes.md, which replaced Signal's cyan):
// green is WrongGPU working (status dots, installed / ready / passed, the end of a progress run) and never changes; the
// accent (Settings > Accent color: Green by default, Blue, Purple, Pink, Cyan, Graphite or the Windows accent) colours the
// controls (links, Back, switches, the chosen radio, Play, Continue, and since the 0.6.0 UI review an update: Update,
// Update N games); red is what needs you (Repair, Clean up, Apply, Uninstall); amber a warning in words; grey neutral. The red / green registration offset (identity: it
// never follows the accent) is drawn in three places only: the mark, the keyboard ring on art and progress runs
// (ProgressRun); the large page titles (PageTitle) are plain since the 0.6.0 UI review.
// The E names are the mockup's CSS names (--text2 is text2, --accent-tint-hover is accentTintHover; "accent" there is the
// red of what needs you). Contrast (WCAG): text2 6.2:1 on a cell and 5.6:1 on the page (light), 7.3 and 8.6 (dark);
// accentText 5.7 / 8.9; white on the accent 5.7 / 5.2; accentText on accentTint 5.4 / 6.0; text3 (disabled labels) 3:1 or
// better on cells, the page and fills. The accent's tokens (tint*) are measured for every accent and both themes by the
// launcher's unit tests (ui_palette.h).
struct Theme {
    bool dark = false;
    Col bg,                // the page (and the toolbar: no bar, the page colour)
        cell,              // a grouped list's cells
        sep,               // hairlines (inset: they start at the text)
        text, text2, text3,  // text; secondary text; disabled labels
        fill, fillHover,   // search, quiet buttons, segmented wells; under the mouse
        segOn,             // a selected segment
        accent, accentText, accentTint, accentTintHover, onAccent,  // red, what needs you: filled, as text, the pills' tint, text on the fill
        tint, tintFill, tintText, tintBg, tintBgHover,               // the accent the player picked: graphics, a fill under white words, words, the pills' tint
        regR, regG,        // the registration offset's red and green (translucent: the mark's and the art ring's; identity, not the accent)
        green, greenText, amber, amberText, greyDot,                // status dots and their words (green: WrongGPU is working, fixed)
        trackOff, trackEdge, knob,  // a switch: the track when off, its edge (3:1 on a cell, WCAG 1.4.11), the knob
        sheet,             // menus, dialogs, the first-run sheet
        material,          // the floating capsule's tint over the blurred page (translucent)
        scrim,             // dims the window under a sheet (translucent)
        focus,             // the keyboard focus ring
        black, white;
    Col artTint, artTintHover, artTintText;  // a red pill on game art (B_ART_TINTED): the same in both themes, art is dark in both
    float cellShadowA = 0.05f;  // a cell's soft shadow (light: 0 1px 2px .05; dark: none)
    float shadowA = 0.18f;      // strength of the floating parts' shadows (menus, the capsule, sheets)
    // a Windows Contrast theme is on (HighContrast): the tokens are the system's colours, and the widgets draw an edge round
    // each control (edge: COLOR_WINDOWTEXT; else sep) instead of telling them apart by fills
    bool hc = false;
    Col edge;
    // The "Bench" names, kept as aliases of the E tokens (SetTheme) while the pages move over; new code uses the E names.
    Col alu, panel, ink, graphite, faint, groove, red, redEdge, onRed, redText, off, hl, under, sel, selInk, rk, knobTop, knobBottom,
        hover, press, well, bezel, brushLight, brushDark, ledRing,
        onArt, onArtDim, artShade, genA, genB, genInk;  // text on game art (white over a dark shade in both themes); generated art
    // the focus ring and the status dots on game art (a banner's dark scrim, the same in both themes): the dark theme's
    // values in both (ArtScope)
    Col focusArt, greenArt, amberArt, greyArt, redArt;
};
extern Theme pal;
void SetTheme(bool dark);
// The accent SetTheme uses from now on (ui_palette.h Accent; windowsRgb: the Windows accent colour, 0xRRGGBB, for A_WINDOWS)
void SetAccent(int accent, unsigned windowsRgb);
int CurrentAccent();
unsigned WindowsAccentRgb();  // the Windows accent colour now (DWM's AccentColor, else its colorization colour)
bool HighContrast();          // a Windows Contrast theme is on (SPI_GETHIGHCONTRAST): SetTheme takes the system's colours
bool TransparencyOn();        // Windows' Transparency effects (off: the floating material and the scrim are opaque, unblurred)
void ContrastPreview(int preset);  // ui-render: a contrast theme's colours without the PC using one (0 off, 1 Night sky, 2 Desert)

// ---- geometry ----------------------------------------------------------------------------------------------------------
struct Rc {
    float x = 0, y = 0, w = 0, h = 0;
    float r() const { return x + w; }
    float b() const { return y + h; }
    float cx() const { return x + w / 2; }
    float cy() const { return y + h / 2; }
    bool has(float px, float py) const { return px >= x && py >= y && px < x + w && py < y + h; }
    Rc in(float d) const { return Rc{x + d, y + d, w - 2 * d, h - 2 * d}; }
    Rc in(float dx, float dy) const { return Rc{x + dx, y + dy, w - 2 * dx, h - 2 * dy}; }
    Rc off(float dx, float dy) const { return Rc{x + dx, y + dy, w, h}; }
};
inline Rc Intersect(const Rc& a, const Rc& b) {
    const float x = std::max(a.x, b.x), y = std::max(a.y, b.y), r = std::min(a.r(), b.r()), bb = std::min(a.b(), b.b());
    return Rc{x, y, std::max(0.f, r - x), std::max(0.f, bb - y)};
}

// ---- images (decoded art: premultiplied BGRA) -------------------------------------------------------------------------
struct Image { int w = 0, h = 0; std::vector<uint8_t> px; };
using ImageP = std::shared_ptr<const Image>;

// ---- drawing -----------------------------------------------------------------------------------------------------------
// Segoe UI Variable Text (body, controls); Segoe UI Variable Display (titles, group headings); both Segoe UI on Windows 10.
// Mono: Cascadia Mono, else Consolas. Icons: Segoe Fluent Icons, else Segoe MDL2 Assets.
enum FontFace { F_TEXT, F_DISPLAY, F_MONO, F_ICON };
struct TS {  // text style
    float size = 13;
    int weight = 400;
    Col color = pal.text;
    int align = 0;         // 0 left, 1 centre, 2 right
    int valign = 0;        // 0 top, 1 centre, 2 bottom
    bool wrap = false;     // false: one line with an ellipsis
    FontFace face = F_TEXT;
    float spacing = 0;     // extra space between letters (DIPs)
    float lineH = 0;       // line height (0: the font's)
    bool underline = false;
    bool trim = false;     // wrapped text: an ellipsis on the last line that fits the box
};
class Gfx {
public:
    ~Gfx();
    bool Init();
    // Live: the window's swap chain (its client size in pixels, at dpi). Offscreen: a WIC bitmap of w x h DIPs at `scale`.
    bool BeginHwnd(HWND hwnd, UINT dpi);
    bool BeginOffscreen(int w, int h, float scale);
    void EndFrame();  // live: present; recreates the device and swap chain when the device is lost
    bool SavePng(const wstring& path);  // the offscreen bitmap
    void Resize(HWND hwnd);
    void DropTarget();
    float scale = 1;
    bool offscreen = false;
    bool occluded = false;  // the last Present said nothing of the window is visible (covered, the session locked): frames wait
    bool dry = false;  // measuring: nothing is drawn (cards measure their height in a dry pass)

    void Clear(Col c);
    void Fill(const Rc& r, Col c);
    void FillRound(const Rc& r, float rad, Col c);
    void FillRound4(const Rc& r, float tl, float tr, float br, float bl, Col c);
    void StrokeRound(const Rc& r, float rad, Col c, float width = 1);
    void Grad(const Rc& r, float rad, Col a, Col b, float x0, float y0, float x1, float y1);  // points in 0..1 of r
    void GradStops(const Rc& r, float rad, const std::vector<std::pair<float, Col>>& stops, float x0, float y0, float x1, float y1,
                   float opacity = 1);  // opacity: the brush's, for a fade (the stops, and so the cached brush, stay the same)
    void Line(float x1, float y1, float x2, float y2, Col c, float width = 1);
    void Circle(float cx, float cy, float rad, Col c);
    void Ring(float cx, float cy, float rad, Col c, float width);
    // covered: an opaque shape is drawn over r next (art, a plate, a sheet), so the layers it hides are not filled at all
    void Shadow(const Rc& r, float rad, float blur, float alpha, float dy = 0, bool covered = false);
    // The offscreen bitmap's pixels (premultiplied BGRA, top-down): the window's icon is drawn this way (MarkIcon)
    bool Pixels(std::vector<uint8_t>& bgra, UINT& w, UINT& h);
    // A blurred, dimmed backdrop under a sheet (Scrim) kept once its entrance ends: while `key` (what the page behind shows)
    // stays the same, the kept picture is drawn instead of copying and blurring the whole window again on every frame.
    bool DrawFrozen(unsigned long long key);
    void Freeze(unsigned long long key);
    // Text. Returns the laid-out height.
    float Text(const wstring& t, const Rc& r, const TS& s);
    float TextW(const wstring& t, const TS& s);
    float TextH(const wstring& t, float width, const TS& s);
    void TextEnd(const wstring& t, float width, const TS& s, float& x, float& y);  // where the last character ends (x) and its line starts (y)
    void Icon(wchar_t glyph, float cx, float cy, float size, Col c);
    // Images: cover (CSS background-size: cover) clipped to a rounded rectangle; contain at an anchor.
    void ImageCover(const ImageP& img, const Rc& r, float rad, float focusY = 0.5f, float opacity = 1, bool topOnly = false);
    void ImageContain(const ImageP& img, const Rc& box, int ax, int ay, float opacity = 1, float shadow = 0);  // ax/ay: 0 start, 1 centre, 2 end; shadow: blur (DIPs)
    // The whole image stretched over r nearest-neighbour (pixel-stepped: the upscale wipe's render-size half), clipped to
    // r's rounded shape.
    void ImageNearest(const ImageP& img, const Rc& r, float rad, float opacity = 1);
    // Text whose first `leadLen` characters have their own colour and weight (e.g. a green "Ready." before the description).
    float TextLead(const wstring& t, size_t leadLen, Col leadC, int leadWeight, const Rc& r, const TS& s);
    // A soft elliptical glow (radial gradient from c to transparent).
    void Glow(float cx, float cy, float rx, float ry, Col c);
    void PushClip(const Rc& r);
    void PopClip();
    void PushRoundClip(const Rc& r, float rad);
    void PopRoundClip();
    // Everything drawn until PopFade at `opacity` (an entrance: a menu, a dialog, a page). ClearType-safe in the live window.
    void PushFade(float opacity);
    void PopFade();
    // A backdrop blur: what is drawn so far under `r` (drawing DIPs) is copied, blurred (a Direct2D Gaussian blur of `blur`
    // DIPs standard deviation, the CSS blur() radius) and drawn back inside r's rounded shape at `opacity`. Call it with no
    // clip or layer open (the copy needs none: draw it before PushFade and pass the fade as `opacity`). False when nothing
    // was drawn (a clip or layer open, no device context, the copy failed): the caller then uses a plain, more opaque
    // material. A dry pass returns true and draws nothing.
    bool Backdrop(const Rc& r, float rad, float blur, float opacity = 1);
    void SetOffset(float dx, float dy);  // content scroll: everything drawn after this is moved
    float OffsetY() const { return oy_; }  // the vertical offset SetOffset set (the Library's scroll anchoring moves it)
    ID2D1RenderTarget* rt = nullptr;
    IDWriteFactory* dw = nullptr;
    IWICImagingFactory* wic = nullptr;
    bool iconFont = true;

private:
    ID2D1Factory1* d2d_ = nullptr;
    // live: a Direct3D 11 device and a flip-model swap chain on the window, drawn through a Direct2D device context
    ID3D11Device* d3d_ = nullptr;
    ID2D1Device* d2dDev_ = nullptr;
    ID2D1DeviceContext* liveDc_ = nullptr;
    IDXGISwapChain1* swap_ = nullptr;
    ID2D1Bitmap1* backBmp_ = nullptr;
    ID2D1DeviceContext* dc_ = nullptr;
    IWICBitmap* wicBmp_ = nullptr;
    ID2D1RenderTarget* wicRt_ = nullptr;
    ID2D1SolidColorBrush* brush_ = nullptr;
    // the images' bitmaps, made once per target; when there are many, those not drawn in the current frame are let go
    // (a big library scrolled end to end, the upscale wipe's small copies)
    // img: the image, not kept alive by this cache (the art cache's limits decide what stays decoded, ui_art.cpp): a bitmap
    // whose image is gone is let go with it (Trim), and one found under a reused address is made again
    // shadow: a logo's drop shadow (ImageContain), one effect per bitmap with its output cached by Direct2D (made again
    // only when its blur or strength changes, not on every frame a scroll or a hover moves it)
    struct Bmp { std::weak_ptr<const Image> img; ID2D1Bitmap* b = nullptr; unsigned long long used = 0; ID2D1Effect* shadow = nullptr; float shadowSd = -1, shadowA = -1; };
    std::map<const Image*, Bmp> bitmaps_;
    unsigned long long frame_ = 0;
    // linear gradient brushes by their stops, made once per target (a draw moves a brush's end points)
    std::map<std::vector<float>, ID2D1LinearGradientBrush*> grads_;
    std::map<std::tuple<int, int, int>, IDWriteTextFormat*> formats_;
    // Made once and kept (SHIP-BAR 1: nothing created per frame): text layouts by their text, style and box (DirectWrite:
    // they outlive a target), an ellipsis sign per text format, rounded paths by their shape (factory resources), the
    // backdrop's copy and blur, the glows' brushes. Those not used for a while are let go (BeginHwnd / BeginOffscreen).
    // lead: TextLead's (its lead's drawing effect is the target's brush, taken off when the target goes: Release)
    struct CachedLayout { IDWriteTextLayout* l = nullptr; float dx = 0; DWRITE_TEXT_METRICS m{}; unsigned long long used = 0; bool lead = false; };
    std::map<std::tuple<wstring, int, int, int, int, int, int>, CachedLayout> layouts_;
    std::map<IDWriteTextFormat*, IDWriteInlineObject*> signs_;
    struct CachedPath { ID2D1PathGeometry* p = nullptr; unsigned long long used = 0; };
    std::map<std::tuple<int, int, int, int, int, int, int, int>, CachedPath> paths_;
    ID2D1Bitmap1* backdropBmp_[2] = {};
    int backdropNext_ = 0;
    ID2D1Effect* blur_ = nullptr;
    ID2D1Bitmap1* frozen_ = nullptr;
    unsigned long long frozenKey_ = 0;
    // the frame that last drew the frozen picture or a backdrop copy: one a frame did not use is let go as it ends (EndFrame;
    // a closed dialog's full-window copies are not kept for the session)
    unsigned long long frozenUsed_ = 0, backdropUsed_[2] = {};
    ID2D1SolidColorBrush* leadBrush_ = nullptr;
    std::map<std::tuple<int, int, int, int>, ID2D1RadialGradientBrush*> glows_;
    const CachedLayout* Cached(const wstring& t, float w, float h, const TS& s, bool balance);
    void Trim();
    wstring families_[4];
    DWRITE_FONT_STRETCH stretch_[4]{DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STRETCH_NORMAL, DWRITE_FONT_STRETCH_NORMAL};
    void Release();
    void TargetReady();
    bool CreateSwapChain(HWND hwnd, UINT w, UINT h, UINT dpi);
    bool BindBackBuffer(UINT dpi);
    bool ResizeSwapChain(UINT w, UINT h, UINT dpi);
    ID2D1SolidColorBrush* B(Col c);
    IDWriteTextFormat* Format(FontFace f, float size, int weight);
    IDWriteTextLayout* Layout(const wstring& t, float w, float h, const TS& s);
    ID2D1Bitmap* Bitmap(const ImageP& img);
    ID2D1PathGeometry* RoundPath(const Rc& r, float tl, float tr, float br, float bl);
    std::vector<ID2D1Layer*> layers_;
    int clips_ = 0;  // axis-aligned clips open (PushClip)
    float ox_ = 0, oy_ = 0;
    void PushMask(const Rc& r, float rad, float opacity);  // a rounded mask layer at `opacity` (popped with PopRoundClip)
};

// ---- art (ui_art.cpp) ----------------------------------------------------------------------------------------------------
// hero: the wide library_hero.jpg at a tile's size and logo: the game's logo.png (alpha) at the banner's (the Library's tiles
// and banner, the game page's banner); header: header.jpg (a tile without a hero: it has the title on it); cover: the
// portrait capsule (library_600x900.jpg, else library_capsule.jpg) for the compact header, the game page without a hero and
// the Not supported rows
struct Art { bool done = false; ImageP cover, hero, logo, header; };
void ArtStart(HWND notify, UINT msg);            // the worker; `msg` is posted when art arrives
Art ArtFor(const wstring& appid);                // what is decoded so far (queues the rest)
ImageP ArtBig(const wstring& appid, bool* done = nullptr);  // the hero at a banner's size (queued when not decoded yet: *done false)
bool HeroOnDisk(const wstring& appid);           // the Steam cache has the game's hero, decoded yet or not (checked once, kept)
void ArtScale(float scale);                      // the window's DPI scale (art is decoded at its drawn size times it)
void ArtLoadNow(const std::vector<wstring>& appids);  // offscreen renders: decoded synchronously
void ArtStop();
// A banner's hero drawn in r: the banner-sized decode (the tile-sized one while it decodes), cross-faded in when it arrives
// instead of popping; `under` draws the banner's fallback (its header or plate) while there is no hero or the first fades
// in over it. Returns the hero drawn (null: none yet); *firstFade: the first hero's opacity while it fades in (its logo
// follows it), else 1.
ImageP HeroCover(Gfx& gx, const wstring& appid, const Rc& r, float rad, float focusY, const std::function<void()>& under, float* firstFade = nullptr, float keep = 1);
bool HeroFading();                                // a banner's hero is fading in (the window keeps drawing frames)

// ---- widgets and the view (ui_widgets.cpp) -----------------------------------------------------------------------------
// R_ITEM with a state (0/1): a tab (the toolbar's Library | Settings); R_SCROLL: the scroll thumb (the mouse only: UI Automation
// scrolls through the window's Scroll pattern)
// R_DIALOG, R_MENU: an open dialog's sheet and an open menu's sheet (UI Automation's Window and Menu, taking no click);
// R_PROGRESS: a progress bar (state: 0..100, -1 while it has no percentage). R_TEXT with a state >= 1: a heading of that level.
enum Role { R_BUTTON, R_TOGGLE, R_RADIO, R_ITEM, R_COMBO, R_LINK, R_MENUITEM, R_EDIT, R_TEXT, R_SCROLL, R_DIALOG, R_MENU, R_PROGRESS };
inline bool Passive(Role r) { return r == R_TEXT || r == R_DIALOG || r == R_MENU || r == R_PROGRESS; }  // read, never pressed
struct Hit {
    int id = 0;
    Rc r;                  // window DIPs (after scrolling, clipped to the view)
    Role role = R_BUTTON;
    wstring name;
    bool enabled = true, focusable = true;
    bool shown = true;     // false: scrolled out of view (r unclipped); kept so Tab and UI Automation reach it
    int state = -1;        // toggle / radio: 1 on, 0 off
    wstring value;         // combo: the shown value
    wstring help;          // UI Automation's help text (a row's sentence under a switch or a segment)
    Rc full;               // window DIPs after scrolling, not clipped to the view
    float scroll = 0;      // the page scroll it was drawn at (focus scrolls into view from the scroll it is going to)
    bool sticky = false;   // in the game page's right column (View::stickCol): it does not move 1:1 with the scroll
};
enum Page { P_LIBRARY, P_GAME, P_SETTINGS, P_COUNT };
enum Task { T_NONE, T_MODEL, T_SELFINSTALL, T_SELFUNINSTALL, T_FIRSTRUN, T_COLLECT, T_RRDATA };  // View::task (T_RRDATA: the Ray Reconstruction data)
// B_PRIMARY: a filled accent capsule (Play, Apply, Continue). B_SECONDARY: a grey capsule (More, Discard in a sheet).
// B_PLAIN: a grey rounded rectangle, radius 8 (a settings row's buttons: "Choose a DLSS file..."). B_GHOST: no fill until
// hovered, radius 8 (the toolbar-like Rescan / Add game, Discard on the capsule). B_GLASS: a translucent white capsule with
// white words, on art under a dark scrim (the banners' Game settings and More).
// B_TINTED: a pill, accent words on the accent tint: an action repeated on every row of a list, so a page keeps one or
// two filled accent buttons (Apple HIG, Buttons: "Keep the number of prominent buttons to one or two per view").
// B_LINK: the accent's words, no fill until hovered (Show, the first run's "Show details"). B_GO: a filled capsule in the
// accent (its fill under white words): Play, Continue, a dialog's default button that is not destructive (the red
// B_PRIMARY is for what needs you). B_ART_TINTED: B_TINTED on art under a dark scrim (the game page's banner: "Update to
// 0.5.5" beside Play), the dark theme's red pill in both themes.
enum BtnKind { B_PRIMARY, B_SECONDARY, B_GHOST, B_GLASS, B_TINTED, B_PLAIN, B_LINK, B_GO, B_ART_TINTED };
// help: what the item does, a second line under its name and its help text for UI Automation (the output filter's choices)
struct MenuItem { wstring text; int id = 0; bool enabled = true, danger = false, sep = false, checked = false; wchar_t icon = 0; wstring help; };
struct Menu {
    bool open = false;
    int owner = 0;          // the widget that opened it
    Rc anchor;              // window DIPs
    bool right = false;     // aligned to the anchor's right edge
    float width = 220;
    std::vector<MenuItem> items;
    int hot = -1;
    ULONGLONG at = 0;       // when it opened (its entrance)
    float scroll = -1;      // a list longer than the room it has scrolls (-1: not placed yet, the checked item is brought into view)
};
// primary: the default button, in the app's tint (B_GO); red (B_PRIMARY) when it is destructive (danger) or an action that
// needs you (attention: Apply)
struct DialogButton { wstring text; int result = 0; bool primary = false, danger = false, attention = false; };
struct Dialog {
    bool open = false;
    int icon = 0;           // 0 question, 1 warning, 2 information, 3 danger, 4 error (kIconError)
    wstring title, body;
    std::vector<DialogButton> buttons;
    int result = -1, cancel = 0;
    ULONGLONG at = 0;       // when it opened (its entrance)
    float scroll = 0, maxScroll = 0;  // a text longer than the window has room for scrolls (the wheel, Up / Down)
};
// A failure in words (an operation or a task that didn't finish, a game that couldn't start): its glyph red (IDENTITY.md:
// red is errors in words, amber a warning), and as a note its Close is the default (Return closes it, as a note's has it)
constexpr int kIconError = 4;
struct Toast { wstring text; int tone = 0; ULONGLONG until = 0, at = 0; };  // tone 0 neutral, 1 good, 2 bad; at: when it was shown
// r: the field's text area (window DIPs); view: the part of the window its page shows content in (a field outside it is
// hidden: under the game page's pinned plate or its unsaved bar)
// name: the field's accessible name, given to the native EDIT (UI Automation reads it there, where the keys go)
// enabled: false while its row is locked (FieldBox draws the text dimmed; the EDIT stays hidden: no keys, no focus)
struct EditBox { HWND h{}; Rc r, view; bool want = false; wstring cue, name; bool enabled = true; };

// Widget ids (ranges)
enum : int {
    ID_NAV_LIBRARY = 10, ID_NAV_SETTINGS, ID_RESCAN, ID_ADDGAME, ID_UPDATEALL, ID_SEARCH, ID_SEARCHCLEAR,
    ID_E_ADDGAME = 30, ID_E_SETTINGS, ID_E_CLEAR,  // the Library's empty states
    ID_ATTNMORE,  // the Library's "Show 4 more" / "Show fewer" under the "Needs attention" group
    ID_STATUSLINE,  // the toolbar's status line (text for UI Automation, not focusable)
    ID_BRAND,  // the toolbar's wordmark (text for UI Automation: the words it shows, WrongGPU or RightGPU on a healthy game's page)
    ID_SHELF0 = 40,  // + 0..3: the Library's shelf headings (text for UI Automation, not focusable)
    ID_FEAT0 = 50,   // the Library's featured banner: + 0 its game's name (text), + 1 Play / Repair / Update, + 2 Game settings
    ID_S_ACCENT0 = 4040,  // + 0..6 (A_COUNT): Settings' accent colour swatches (a radio group)
    ID_TILE = 1000000, ID_TILEQA = 2000000,  // + game index (kTileSpan games: above every other id)
    ID_G_BACK = 3000, ID_G_PRIMARY, ID_G_SECOND, ID_G_PLAY, ID_G_MORE, ID_G_MASTER,
    ID_G_LEARN,  // Learn more: the public page of a tool of the player's the game needs (REFramework, gamerules.h), in the browser
    ID_G_RECHECK,  // Check again: that game's folder read again for the tool (its callout)
    ID_G_EXEC0 = 3010, ID_G_FILTER = 3020,
    ID_G_PRE0 = 3030, ID_G_SPARKLE = 3040, ID_G_FPS0 = 3050, ID_G_DETAILS = 3060, ID_G_LOG, ID_G_COLLECT, ID_G_FOLDER, ID_G_KEEP,
    ID_G_REVERT, ID_G_OKEEP, ID_G_OREVERT, ID_G_DISCARD, ID_G_APPLY, ID_G_LOCKED, ID_G_NOTICE, ID_G_CLEAN, ID_G_ADOPT, ID_G_FOUND, ID_G_OPTIMORE, ID_G_ADVMORE, ID_G_GUIDEFPS,
    ID_G_TITLE, ID_G_DIRTEXT, ID_G_EXETEXT,  // text for UI Automation (not focusable): the game's whole name, its folder, its exe
    ID_G_COVER,  // the game page's cover: not a hit, the name of its upscale wipe (UpscaleWipe)
    ID_G_STATUS, ID_G_PROGRESS, ID_G_UNSAVED,  // text for UI Automation: the header's status (or step), its progress, the capsule's words
    // Ray Reconstruction (rr.h): its switch, the data's Convert, Repair (an install without the RR files); its quality's
    // segments + 0 Exact, 1 Accurate, 2 Fast (Exact is drawn only with Settings' Exact mode, or while the game is set to it)
    ID_G_RRON = 3086, ID_G_RRCONVERT, ID_G_RRREPAIR, ID_G_RRPREP, ID_G_RRMODE0 = 3090,  // ID_G_RRPREP: Prepare (rrprep.h)
    // Borderlands 4 / Hogwarts Legacy's Engine.ini setting (UI review #2): its segments (+ 0 WrongGPU, 1 Myself) and the steps'
    // Open folder, Copy and Check again; Fix it again (WrongGPU's line gone or the file no longer read-only: r2 #6)
    ID_G_RRCFG0 = 3093, ID_G_RRCFG_OPEN = 3095, ID_G_RRCFG_COPY, ID_G_RRCFG_CHECK, ID_G_RRCFG_FIX,
    ID_G_OPTI0 = 3100,  // + editable OptiScaler row (x10: + option)
    ID_G_OPTI_END = ID_G_OPTI0 + 10 * 64,  // room for 64 rows; the Settings ids start above it
    ID_S_MODELFILE = 4000, ID_S_RECONVERT, ID_S_OPTIFILE, ID_S_OPTIDIR, ID_S_ADDFOLDER, ID_S_CLEARFOLDERS, ID_S_PLAY, ID_S_UPDATE,
    ID_S_URL, ID_S_CHECKNOW, ID_S_SAVEURL, ID_S_SELFINSTALL, ID_S_SELFUNINSTALL,
    ID_S_MODELTEXT, ID_S_HIPTEXT,  // text for UI Automation and a tooltip (not focusable): the model data's and the HIP runtime's files
    ID_S_MODELMORE,  // a ready model data row's "Options..." (a menu: Choose a DLSS file..., Convert again)
    ID_S_OPTITEXT, ID_S_DATATEXT,  // text for UI Automation and a tooltip (not focusable): the OptiScaler download's and the launcher data's whole paths
    ID_S_THEME0 = 4020,  // + 0 system, 1 light, 2 dark
    ID_S_INDEX0 = 4030,  // + 0..2: Settings' section index (Model and runtime, Games, Launcher)
    // Settings > Ray Reconstruction data: Options... (a menu: Choose a DLSS file..., Convert again), the two ways, its words for UI
    // Automation (not focusable); Exact mode's switch
    ID_S_RRMORE = 4050, ID_S_RRFILE, ID_S_RRCONVERT, ID_S_RRTEXT, ID_S_EXACT,
    ID_S_REMOVEDATA = 4070,  // a portable copy's Launcher row: Remove data... (the shared data, its way out: it has no uninstall)
    ID_S_FOLDERRM0 = 4100,  // + index: an added game folder's Remove (kMaxFolderRows of them)
    ID_S_FOLDERTEXT0 = 4400,  // + index: an added game folder's words for UI Automation and its tooltip (its whole path; not focusable)
    ID_MENU = 5000, ID_MENUBOX = ID_MENU + 100,  // + item; the open menu's sheet (UI Automation's Menu)
    ID_DIALOG = 6000, ID_DIALOGBOX = ID_DIALOG + 98, ID_DIALOGBODY,  // + button; the open dialog's sheet (its title), its text
    ID_FR_MODEL = 7000, ID_FR_INSTALL, ID_FR_PORTABLE, ID_FR_CONTINUE, ID_FR_CHECKS, ID_FR_SEARCH,
    ID_FR_SHEET,  // the first run's sheet (UI Automation's Window, a dialog: PaintAll adds it)
    ID_SCROLLBAR = 8000,
    ID_CAP_MIN = 8010, ID_CAP_MAX, ID_CAP_CLOSE,  // the caption buttons drawn in the toolbar (the window has no system title bar)
    ID_AUTO0 = 9000,  // + n: words for UI Automation (AddText, AddProgress) numbered as a frame draws them
    // menu commands
    M_OPEN_FOLDER = 100, M_OPEN_LOG, M_COLLECT, M_REPAIR, M_ROLLBACK, M_UNINSTALL, M_RESET, M_UPDATE, M_INSTALL, M_CLEAN, M_ADOPT,
    M_REMOVEGAME,  // a game added by hand taken out of the Library (its page's More)
    M_FILTER0 = 200, M_OPTI0 = 300,
};
constexpr int kTileSpan = ID_TILEQA - ID_TILE;  // the Library draws at most this many games (ui_library.cpp)
constexpr int kMaxFolderRows = 256;             // Settings lists at most this many added game folders, each with its Remove

// What a game's page shows of WrongGPU's OptiScaler setup (FillOpti, or a fixture).
struct OptiView {
    bool have = false;      // WrongGPU's OptiScaler setup is installed in this game
    wstring headline;       // "OptiScaler 0.9.0 as dxgi.dll (...)"
    wstring version, proxy, proxyReason, folder;
    std::vector<OptiRow> editable, locked;
    std::vector<OptiOutside> outside;
    string iniText;         // for the dump's opti_info
};

struct FirstRun {
    bool open = false, busy = false, installOffer = false;
    int choice = 0;                               // the install offer: 0 none yet, 1 install for this user, 2 portable
    bool checksOpen = false;                      // the checks open on a short summary; true: every check shown (Show details)
    std::vector<std::pair<int, wstring>> lines;  // 0 ok, 1 note, 2 problem
    float scroll = 0, maxScroll = 0;              // the card scrolls when the window is too short for it
    float bodyTop = 0, bodyBottom = 0;            // the part that scrolls (window DIPs): the base plate under it stays put
    ULONGLONG at = 0;                             // when it opened (its entrance)
    // the sheet's height eases to a new one (checks arriving, Show details, a problem's buttons) instead of jumping:
    // from, to (DIPs), when it started (0: not moving), and the entrance it was sized for (a new one starts at its height)
    float fromH = 0, toH = 0;
    ULONGLONG resizeAt = 0, sizedAt = 0;
};

struct View {
    HWND hwnd{};
    bool live = false;           // a window on screen (not ui-dump / ui-render)
    float W = 1180, H = 760;     // client size in DIPs
    Page page = P_LIBRARY;
    ULONGLONG pageChangedAt = 0;  // a double click right after a page change is ignored
    std::vector<Hit> hits;
    std::map<int, wstring> tips;  // this frame's hover tooltips by hit id (an icon-only button, a name cut off)
    int hot = 0, down = 0, focus = 0;
    bool focusVisible = false;
    float scroll[P_COUNT]{}, target[P_COUNT]{}, contentH[P_COUNT]{};
    float pinH = 0;              // the game page's plate pinned under the window's top edge: its height (0: not pinned)
    int heroTall = -1;           // the game page's art window, decided when the page opens: 1 full height, 0 short, -1 not yet
    bool dragThumb = false; float dragY = 0, dragScroll = 0;
    // the overlay scroll thumb shows while the page scrolls and while the pointer is over its track, then fades (Windows 11):
    // when it was last wanted (NowMs), and where the thumb was drawn (window DIPs: a click on the track beside it pages)
    ULONGLONG scrollSeenAt = 0;
    Rc thumbRc;
    bool active = true, maximized = false;  // the window: active (the caption buttons' glyphs dim when not), maximized (Restore)
    std::map<int, float> anim;   // hover lift of the tiles (0..1)
    Menu menu;
    Dialog dialog;
    Toast toast;
    FirstRun first;
    wstring search;
    EditBox edSearch, edUrl;
    std::vector<EditBox> edOpti;  // one per editable OptiScaler number field
    // game list
    std::vector<GameEntry> games;
    bool scanning = false;
    bool listCached = false;     // v.games is the last session's list (LoadLibrarySnapshot) until the start-up scan's lands
    bool attnAll = false;        // the Library's "Needs attention" group shown whole (else its most urgent few and "Show 4 more")
    bool rackEmpty = false;      // the Library shows no bay, only an empty state (its scanning light steps: Animating)
    // the game page
    int curIndex = -1;
    GameEntry cur;
    bool pageReading = false;    // the page shows before its files are read (OpenGame on a slow disk): header only, settings wait
    Profile saved, edit;          // the game's profile and the values shown (Apply writes edit)
    std::vector<OutsideEdit> outside;
    OptiView opti;
    std::vector<string> optiValues;  // shown values of opti.editable
    // an OptiScaler value Apply refused (OnApply): the game, its row, the text it had and why; its field says it is invalid (a
    // red edge, UI Automation's IsDataValidForForm and the reason as help text) until that text changes (FieldInvalidWhy)
    struct OptiBad { wstring dir; size_t row = (size_t)-1; string value; wstring why; } optiBad;
    bool detailsOpen = false, lockedOpen = false, foundOpen = false, optiMore = false;
    bool advOpen = false;  // the game page's Upscaler: its Advanced rows shown (UI review #5; AdvancedWanted when a page opens)
    int rrcfg = -2; wstring rrcfgFile;  // the game's Engine.ini line for Ray Reconstruction (PageData::rrcfg; UI review #2)
    wstring rrcfgWhy;  // rrcfg 3: why Engine.ini couldn't be read (r3 #8)
    bool rrcfgFixed = false;  // the install record has WrongGPU's line (PageData::rrcfgFixed; r2b #1: else "on" waits for the folder)
    // "Show me how" (ui_rrconfig.cpp): the game whose page scrolls to its Engine.ini steps, with the focus on Open folder, once
    // they show there (after the Apply that sets rr_config_fix=off); rrcfgCardY: the card's top in this frame (content DIPs)
    wstring rrcfgReveal; float rrcfgCardY = -1;
    // "Show me how" (r3 #6): Myself put on this game's page as an unsaved choice when its page shows (ShowEntryWith, once), so
    // the steps show without a permission prompt and even when saving the choice fails; Apply saves it
    wstring rrcfgMyselfDir;
    bool paintScrolled = false;  // a paint moved the page's scroll on purpose (RevealRrConfigCard): RenderFrame's measure pass keeps it
    wstring notice, noticeDir; int noticeTone = 0;  // a failed operation's note: on the page of the game it ran on (noticeDir) only
    // operations
    bool busy = false; wstring busyDir, step; float progress = 0;
    wstring busyVerb;            // what the running operation does, in words ("Updating…"), set when it starts
    std::vector<OpRequest> queue; int queueTotal = 0, queueDone = 0;
    std::vector<wstring> queueDirs;  // every game of the running queue (Update all): they keep their place in the Library until it ends
    std::vector<wstring> queueComing;  // Update all preparing (each game read, then asked): every game it will update (PlayBlocked)
    bool rescanPending = false;  // the settings changed while a scan ran (a game or folder added): one more scan when it ends
    bool closeAfterOp = false;   // the window was closed with Apply chosen for unsaved changes: it closes once that Apply ends
    // A long job that is not a game operation (converting the model, installing or uninstalling the launcher itself) runs on a
    // worker thread with v.busy set (no operation starts meanwhile and the window cannot close half way): which one it is
    // (its button says so: "Converting..."); T_NONE none
    int task = 0;
    // The values an Apply is writing (its game, the profile as the player left it and as it was, the OptiScaler rows as they
    // were and the values the player typed): if it fails, they are shown again as unsaved changes instead of being lost
    struct PendingApply { wstring dir; Profile edit, saved; std::vector<OptiRow> rows; std::vector<string> opti; } pendingApply;
    Rc toastRc;                  // where the toast was drawn (window DIPs): it fades away while the pointer is over it
    // settings page
    wstring urlText;
    bool checkingUpdate = false;  // "Check now" is waiting for the release address (a worker thread)
    // health (read off the UI thread, and again after a conversion or an operation: never per frame)
    bool modelReady = false; wstring gpuName, executorLine;
    wstring modelWhy, modelSource;  // why the model data is not ready; the DLSS file it was converted from
    wstring hipLine;                // the HIP runtime Settings shows (empty: not found)
    // Ray Reconstruction (rr.h): the RR data's state (read with the model's, off the UI thread), and whether this build has the
    // RR files (rrBuild; a game page and Settings show nothing of RR without them, unless a game already has them)
    bool rrReady = false, rrBuild = false; wstring rrWhy, rrSource;
    // WrongGPU's edition of OptiScaler is what installs use (optiedition.h, downloaded): without it a game whose RR needs it (gamerules.h) can't
    // use WrongGPU's Ray Reconstruction, and its page says so
    bool rrEdition = true;
    bool launcherInstalled = false;
    float pageCx = 0;            // the centre of the page's content (toasts are centred on it)
    int editFocusId = 0;         // a text field the keyboard went to before it was in view: it takes the keys once shown
    bool fixture = false;        // ui-render with fixture data (no game folders are read)
    wstring fixtureHip;          // ui-render: the HIP runtime line Settings shows (empty: not found)
    wstring fixtureRrPrepared;   // ui-render: the game (its exe) whose RR kernels count as prepared, as an install's prepare
                                 // leaves them (the render has no rr-prepare\prepared.txt); empty: none (r5 #8)
    bool fixturePackaged = false;  // ui-render: Settings as if started inside an app package
    float motionAt = -1;         // ui-render: every entrance drawn at this point of its motion (0..1); -1: live timing
    // the upscale wipe (UpscaleWipe): where the pointer is (window DIPs; -1 outside the window); the art drawn this frame
    // that can wipe (window DIPs, clipped to the view; owner: the hit the pointer must be on, a lockup's, 0 for the game
    // page's cover); the art the pointer wipes (0: none); when each wipe started (its fade)
    float mouseX = -1, mouseY = -1;
    struct WipeArt { int id = 0, owner = 0; Rc r; };
    std::vector<WipeArt> wipeArts;
    int wipeHot = 0;
    std::map<int, ULONGLONG> wipeAt;
    // a progress bar's new amount was gliding in the last drawn frame, its light or segment travels in the active window, or
    // RightGPU's flip moves (Animating keeps the frames coming at the display's rate; an inactive window's bar steps on timer 9)
    bool barMoving = false;
    // RightGPU's flip (BrandNow): the game whose install, repair or update just finished on its page, and when; flipT: ui-render's
    // point of the flip (0..1, -1: live timing)
    wstring flipDir; ULONGLONG flipAt = 0; float flipT = -1;
    // the Library's featured banner: the game it shows (index into games, -1 none); ui-render: art left out per app id
    // (1 the hero and logo, 2 the header, 4 the cover), to draw every fallback
    int featIndex = -1;
    // the game page's right column (Last run, Game folder; ui_game.cpp Cards), whose hits are `sticky`: its top (content
    // DIPs) at scroll s is y while s <= pinAt, else max(y, min(s + stick, max)) (it stops under the pinned header). Reveal
    // brings its controls into view by where the column will be, not by the scroll alone.
    struct StickyCol { bool on = false; float y = 0, stick = 0, max = 0, pinAt = 0, pinH = 0; } stickCol;
    std::map<wstring, int> artHide;
    std::map<int, ULONGLONG> wipeOut;  // a wipe the pointer (or the focus) left: when its fade out started (as long as it came in)
};
extern View v;
// release 0.6.0: Ray Reconstruction's kernels prepared for this game's program name (rrprep.h; cached until the next prepare)
bool RrReadyForGame(const wstring& dir, const wstring& exe, wstring* why = nullptr);
wstring RrBackgroundWords(const wstring& exe);  // 0.6.0 polish: the game page's RR row while it is prepared again in the background
static_assert(ID_G_OPTI_END <= ID_S_MODELFILE, "the OptiScaler rows' ids must not reach the Settings page's ids");
static_assert(ID_G_LEARN < ID_G_EXEC0 && ID_G_RECHECK < ID_G_EXEC0, "the header's ids must stay under Runs on's segments");
static_assert(ID_G_UNSAVED < ID_G_RRON && ID_G_RRREPAIR < ID_G_RRMODE0 && ID_G_RRPREP < ID_G_RRMODE0 && ID_G_RRMODE0 + 3 <= ID_G_RRCFG0 &&
              ID_G_RRCFG0 + 2 <= ID_G_RRCFG_OPEN && ID_G_RRCFG_FIX < ID_G_OPTI0,
              "the game page's fixed ids and Ray Reconstruction's must stay apart and under the OptiScaler rows' ids");
constexpr int kAutoSpan = 1000;  // AddText's ids (a frame draws fewer words than this)
static_assert(ID_MENUBOX < ID_DIALOG && ID_DIALOGBODY < ID_DIALOG + 100 && ID_DIALOG + 100 <= ID_FR_MODEL && ID_SCROLLBAR < ID_AUTO0 && ID_AUTO0 + kAutoSpan <= ID_TILE,
              "the menus', dialogs' and UI Automation words' ids must stay apart and under the game rows' ids");
static_assert(M_REMOVEGAME < M_FILTER0, "the More menu's commands must stay under the output filter's");
static_assert(ID_STATUSLINE < ID_SHELF0 && ID_SHELF0 + 4 <= ID_FEAT0 && ID_FEAT0 + 3 <= ID_G_BACK, "the Library's and the toolbar's ids must stay under the game page's ids");
static_assert(ID_S_DATATEXT < ID_S_THEME0, "the Settings rows' fixed ids must stay under the appearance segments' ids");
static_assert(ID_S_THEME0 + 3 <= ID_S_INDEX0 && ID_S_INDEX0 + 3 <= ID_S_ACCENT0 && ID_S_ACCENT0 + A_COUNT <= ID_S_RRMORE && ID_S_EXACT < ID_S_FOLDERRM0 &&
              ID_S_FOLDERRM0 + kMaxFolderRows <= ID_S_FOLDERTEXT0 && ID_S_FOLDERTEXT0 + kMaxFolderRows <= ID_MENU,
              "Settings' section index and folder ids must stay between its theme segments and the menus");
static_assert(ID_S_EXACT < ID_S_REMOVEDATA && ID_S_REMOVEDATA < ID_S_FOLDERRM0, "Remove data must stay between the RR rows' ids and the folders' ids");
static_assert(ID_FR_SHEET < ID_SCROLLBAR && ID_SCROLLBAR < ID_CAP_MIN && ID_CAP_CLOSE < ID_AUTO0, "the caption buttons' ids must stay between the first run's and the UI Automation words' ids");
static_assert(ID_SCROLLBAR < ID_TILE && ID_TILE + kTileSpan <= ID_TILEQA && (long long)ID_TILEQA + kTileSpan <= 0x7fffffffLL,
              "the games' ids (a lockup, its action) must stay above every other id and apart from each other");

// Hits and states
void AddHit(int id, Rc r, Role role, const wstring& name, bool enabled = true, int state = -1, const wstring& value = L"", bool focusable = true);
// Words the window draws that carry meaning (a status, a read-out, a caption, a dialog's question), for UI Automation and
// the dump: a text hit that takes no focus. id 0: the frame's next ID_AUTO0 + n. heading: its level (0 none). Add it before
// the controls drawn over the same place (the last hit takes the click). Returns its id.
int AddText(const Rc& r, const wstring& text, int id = 0, int heading = 0);
// A progress bar for UI Automation (ProgressBar, RangeValue 0..100): fraction 0..1, or below 0 without a percentage.
void AddProgress(int id, const Rc& r, const wstring& name, float fraction);
void Describe(int id, const wstring& help);  // UI Automation's help text of this frame's hit `id` (a row's sentence)
bool Hot(int id);
bool Down(int id);
bool Focused(int id);
void FocusRing(Gfx& gx, const Rc& r, float rad, int id);
// The page's focus ring where it reaches over the toolbar's bottom edge (a control at the top of the page, "< Library"):
// drawn again outside the page's clip once the page is drawn, so the ring is never cut flat (PaintAll)
void FocusRingOverBar(Gfx& gx);
// Controls drawn on game art while one lives (the featured banner's, the game page banner's): their focus ring and status
// dots take the art's colours (Theme focusArt...), the same in both themes, not the page's (the light theme's dark ring and
// green are under 3:1 on the scrim)
struct ArtScope { ArtScope(); ~ArtScope(); ArtScope(const ArtScope&) = delete; ArtScope& operator=(const ArtScope&) = delete; };
// Content scroll: page painters draw in content coordinates; hits are converted with the current offset and clipped.
struct ContentView { Rc view; float scroll = 0; bool dry = false; };
// Runs `paint` once dry (measuring, no hits) and returns the height it reports.
float Measure(Gfx& gx, const std::function<float()>& paint);
extern ContentView cv;

// ---- the window's frame (option E): a toolbar on the page colour, each page below it in one centred reading column ------
// The toolbar (DrawToolbar): the mark and "WrongGPU" on the left, the Library | Settings segmented nav in the middle (the
// Library segment carries the count of games that need attention), one quiet status line on the right. Pages get the area
// under it (PaintAll: Rc{0, kBarH, W, H - kBarH}); content coordinates are window coordinates at scroll 0, so a page
// starts at area.y and reports its bottom as contentH.
constexpr float kBarH = 52;
// The toolbar is the window's title bar too (no system caption above it): its empty parts drag the window, a double click
// maximizes, and Minimize, Maximize / Restore and Close sit at its right end (46 DIPs each, as Windows 11 draws them).
constexpr float kCapW = 46;
float CaptionW();                // the caption buttons' width (the offscreen renders draw them too, as the window has them)
int CaptionAt(float x, float y); // the caption button at a window point (ID_CAP_*), 0 none
// One column for every page, so a page's left edge and its title never move between pages: the Library's (BE-library:
// min(1080, the window - 96)). The Library sets its shelves to whole lockups inside it; Settings and the game page fill it.
constexpr float kColMax = 1080;  // the reading column (the Library, the game page, Settings)
constexpr float kGutter = 48;    // the least room either side of the column (a small window)
inline float ColW(float areaW) { return std::min(kColMax, areaW - 2 * kGutter); }
inline float ColX(const Rc& area) { return std::round(area.x + (area.w - ColW(area.w)) / 2); }
void DrawToolbar(Gfx& gx);
bool ToolbarHit(int id);         // the toolbar's own hits (nav, status line): not page content
wstring StatusLine();            // "Model M ready, RX 9070 XT, Direct3D 12" (joined by middle dots)
// Busy states block only what conflicts (SHIP-BAR 1). Play, and a game's settings, wait for: the game running, an operation
// in that game or still to come for it in Update all, a task that changes every game or the model data (converting Model M,
// installing or uninstalling the launcher, the first run's install). Operations in other games and collecting logs leave
// them free. An operation (Install, Update, Repair, Apply...) still waits for any other one: one runs at a time.
bool PlayBlocked(const GameEntry& e);
// Why an action of game e waits, in words for its tooltip and UI Automation ("Waits until WrongGPU finishes updating Quarry Nine.");
// empty when nothing runs
wstring BusyWhy(const GameEntry& e);
// The launcher's uninstall (or a portable copy's Remove data...) reading every game's record and the shared data before it asks:
// no game's read, so its busy verb says what it looks at; its own button shows "Preparing...", the others wait in these words
constexpr const wchar_t* kLookVerb = L"Looking at your games\u2026";
inline bool LookingAtGames() { return v.busy && v.task == T_NONE && v.busyDir.empty() && v.busyVerb == kLookVerb; }

// Sizes of option E's parts (DIPs; the mockup's CSS)
constexpr float kGroupR = 12;    // an inset grouped list's corners
constexpr float kSectionGap = 30;  // from one section's list to the next section's heading
constexpr float kHeadGap = 10;   // a group heading to its list (6 more when a caption sits between them: GroupCaption)
constexpr float kRowH = 68;      // a Library row (a 36 x 54 cover)
constexpr float kRowPad = 14;    // a Library row's side padding; its hairline starts at kRowPad + kCoverW + 14 = 64
constexpr float kSRowH = 54;     // a settings row's least height (the game page, Settings): 11 above and below its words
constexpr float kCellPad = 16;   // a settings row's side padding (its hairline starts here; 64 with a glyph tile)
constexpr float kKvH = 44;       // a read-out row ("Upscaler   Model M")
constexpr float kActH = 44;      // an action row ("Open log" with its glyph at the right): the read-out rows' height
constexpr float kMoreH = 46;     // the Library's "Show 4 more" row
constexpr float kCoverW = 36, kCoverH = 54;        // a Library row's cover (radius 6)
constexpr float kBigCoverW = 88, kBigCoverH = 132;  // the game page's cover (radius 10)
constexpr float kSwitchW = 46, kSwitchH = 28;
constexpr float kPillH = 28, kPillMinW = 78;
constexpr float kGhostPad = 12;  // a B_GHOST button's label inset (line its label up with text: x - kGhostPad)

// Widgets (each draws and adds its hit). Every control is 28 DIPs or more; the focus ring is 2 px, 2 DIPs outside.
// rad: -1 the kind's own (capsule h/2; B_PLAIN and B_GHOST 8; B_LINK 6).
float Button(Gfx& gx, int id, float x, float y, const wstring& text, BtnKind kind, wchar_t icon = 0, bool enabled = true, float h = 36, float minW = 0, bool alignRight = false,
             const wstring& name = L"", float rad = -1);
float ButtonW(Gfx& gx, const wstring& text, wchar_t icon, float h, BtnKind kind = B_SECONDARY);
// A row's pill action (B_TINTED, 28 high, at least 78 wide; solid: the filled accent, for the one Play a list leads with),
// its right edge at `right`, centred on `cy`. Returns its width.
float Pill(Gfx& gx, int id, float right, float cy, const wstring& text, bool solid, bool enabled, const wstring& name);
// A switch, 46 x 28 (the accent track when on, a white knob), right-aligned in r and centred on its middle.
void Switch(Gfx& gx, int id, const Rc& r, bool on, bool enabled, const wstring& name);
// Segments 28 high in a 32 DIP well (fill), the selected one raised (segOn). The well is centred in the h + 8 the layouts
// reserve (h under 28 is drawn at 28). Equal widths (Apple HIG) unless counts are given.
// minSeg: the least width of a segment (a list's segmented controls share one, so their left edges line up).
float Segmented(Gfx& gx, int idBase, float x, float y, const std::vector<wstring>& opts, int sel, bool enabled, const wstring& name, float h = 28,
                const std::vector<wstring>& counts = {}, bool alignRight = false, float minSeg = 0);
float SegmentedW(Gfx& gx, const std::vector<wstring>& opts, float minSeg = 0);  // a segmented control's width (equal segments)
float SegmentW(Gfx& gx, const std::vector<wstring>& labels);  // the width of the widest of these labels' segments
// Filter chips (the Library's All 14 / With WrongGPU 8 ...): 30 high capsules, the selected one on the fill. The same hits as
// Segmented (radio items "name: option (count)"). Returns the width. (The name Tabs is kept from the "Bench" look.)
float Tabs(Gfx& gx, int idBase, float x, float y, const std::vector<wstring>& opts, int sel, const wstring& name, const std::vector<wstring>& counts, bool measureOnly = false);
void Select(Gfx& gx, int id, const Rc& r, const wstring& text, bool enabled, const wstring& name);  // a pop-up button on the fill, radius 8
float Chip(Gfx& gx, float x, float cy, const wstring& text, Col fg, Col border, Col bg = Col{0, 0, 0, 0});
float Tag(Gfx& gx, float x, float cy, const wstring& text, Col fg, Col bg);  // a small label on the fill, radius 5 ("+0.11 ms")

// Inset grouped lists. A section: GroupHeading, optionally GroupCaption, then GroupList (the white cell with its rows), then
// optionally GroupFoot; kSectionGap to the next section.
//   y += GroupHeading(gx, x, y, w, L"Needs attention", L"7", trailW) + kHeadGap;
//   y += GroupList(gx, x, y, w, [&](float top) { float yy = top; ...rows, Hairline(gx, x + inset, yy, x + w) between them...; return yy - top; });
// GroupList runs `rows` once dry to measure, draws the cell, then runs it again for real; it returns the height.
float GroupHeading(Gfx& gx, float x, float y, float w, const wstring& title, const wstring& count = L"", float trailW = 0);  // 17 Display; returns 22
float GroupCaption(Gfx& gx, float x, float y, float w, const wstring& text);  // 13 text2, under a heading (wraps); returns its height + 10
float GroupFoot(Gfx& gx, float x, float y, float w, const wstring& text);     // 12.5 text2, under a list, 16 in; returns 8 + its height
float GroupList(Gfx& gx, float x, float y, float w, const std::function<float(float top)>& rows);
void GroupBox(Gfx& gx, const Rc& r, float rad = kGroupR);  // a list's cell (and in the light theme its soft shadow)
void Hairline(Gfx& gx, float x, float y, float right);     // a 1 px separator from x to right (snapped to a device pixel)
float HairlineW(const Gfx& gx);                            // its thickness in DIPs (a line flush with a bar's bottom: y = bottom - it)
// A Contrast theme's edge round a control (the window text colour, the highlight's under the pointer, gray when off);
// nothing in the other themes. For controls drawn outside the widgets (a lockup's pill).
void HcEdge(Gfx& gx, const Rc& r, float rad, bool hot = false, bool enabled = true);
void ArtHcEdge(Gfx& gx, const Rc& r, float rad, bool hot = false, bool enabled = true);  // the same on game art (its text colour)
// A row's hover / press fill inside a group (the fill at 45 %), rounded where the row meets the group's corners.
void RowFill(Gfx& gx, const Rc& row, bool first, bool last, bool pressed = false);
void StatusDot(Gfx& gx, float cx, float cy, Col c);        // the 7 DIP status dot (green / amber / greyDot)
// An update ready, where a state's dot would be: an up arrow in the line's own colour (`words`), not a dot (green is "worked",
// amber "needs you"; the accent's dot was the same green by default: UI review r2 #11)
void UpdateMark(Gfx& gx, float cx, float cy, Col words);
void Chevron(Gfx& gx, float cx, float cy);                 // a row's ">" (text2)
// A game's cover: its portrait art from the Steam cache (Art::cover), else a generated plate (CoverPlate). `shadow`: the
// small row shadow (rows) or the deeper art shadow (the game page's cover, kBigCoverW).
void Cover(Gfx& gx, const Rc& r, const wstring& appid, const wstring& name, float rad = 6, bool shadow = true);
// The upscale wipe (IDENTITY.md, "The moment") on art under the pointer or the keyboard's focus: the left half shows the art
// at the game's render size (nearest-neighbour from a small copy, about a 3.5th of the size, made once per image), a 2 DIP
// white split line with a soft shadow, light capsules with the input and output heights ("960p" / "1440p": the last run's
// sizes when it recorded them). Fades in over kWipeMs (at once with Animation effects off). id: the art's own id (a
// lockup's, ID_G_COVER); owner: the hit the pointer must be on (0: the art has none); r: content DIPs; img: the art drawn
// there (cover-fit at focusY, as ImageCover draws it; keep < 1: as HeroCover's keep crops it). Returns how far it shows (0..1):
// 0 without Steam art (a plate has nothing to upscale). A logo drawn after it stays crisp above it. marks false: the split
// line and the capsules are left to WipeMarks, for art that has a scrim over it (drawn after the scrim, they stay white).
float UpscaleWipe(Gfx& gx, int id, int owner, const Rc& r, float rad, const ImageP& img, float focusY, const LastRun& last, bool focused, float keep = 1,
                  bool marks = true);
void WipeMarks(Gfx& gx, const Rc& r, float t, const LastRun& last);  // the wipe's split line and capsules at `t` (UpscaleWipe's result)
bool WipeSizes(const LastRun& last);  // the last run recorded both sizes: the wipe shows its capsules
int WipeArtAt(float x, float y);  // the art whose wipe the pointer at (x, y) shows (window DIPs, the last frame's art); 0 none
constexpr float kWipeMs = 150;
// A progress run (the fourth registration place): red at its start to the working green at its end, rounded, over r.
void ProgressRun(Gfx& gx, const Rc& r);
// A progress bar on the fill (the Library's lockups, the game page's header and compact header: one pattern): fraction > 0,
// the run to it, a new amount reached by a short glide (a step's jump is not a leap) and a soft light moving along the run
// while a long step works; fraction <= 0 (no amount yet), a segment travelling along the track, never a stub that reads as
// "3 % done". Animation effects off: the amount at once, no light; without an amount the whole run dimmed. active false:
// the track alone (a game waiting in Update all).
void ProgressBar(Gfx& gx, const Rc& bar, float fraction, bool active = true);
// A small activity indicator (macOS's: spokes fading round a circle), radius r: something works and has no amount to show
// (the first run's checks, a conversion). It turns with Animation effects off too (status, not decoration).
void BusyIndicator(Gfx& gx, float cx, float cy, float r, Col c);
// A large page title (Library, Settings, Welcome to WrongGPU): drawn at -1.5 DIP in registration red and at +1.5 DIP in
// registration green, then in its own colour on top. Never game titles, headings or body text.
void PageTitle(Gfx& gx, const wstring& text, const Rc& r, const TS& s);
void CoverPlate(Gfx& gx, const Rc& r, const wstring& name, float rad);  // a quiet two-stop gradient from the name, its initials
wstring Monogram(const wstring& name);                   // "Alan Wake 2" -> "AW", "The Witcher 3" -> "W3"
void Card(Gfx& gx, const Rc& r, float rad = kGroupR);     // = GroupBox (the "Bench" name)
// A sheet lifted over the page (menus, dialogs, the first-run sheet): a soft far shadow, the sheet colour, a hairline edge.
void Sheet(Gfx& gx, const Rc& r, float rad, float lift = 1);
// The floating capsule's material (the unsaved-changes capsule, toasts): its shadow, the page under it blurred (Backdrop),
// the translucent material tint and a hairline edge. An opaque sheet when the blur is not available or not wanted (Windows'
// Transparency effects off, a Contrast theme).
void Material(Gfx& gx, const Rc& r, float rad, float blur = 18, float opacity = 1);
// The window behind a sheet (the first run, dialogs): blurred (6 DIPs at amount 1) and dimmed with the scrim. amount 0..1
// (an entrance).
void Scrim(Gfx& gx, float amount);
void BasePlate(Gfx& gx, const Rc& sheet, float h, float rad);  // a hairline over a sheet's button row
Col PageText();  // secondary words drawn on the page itself (not on a cell): text2 (5.6:1 light, 7.4:1 dark)
// A section label over a group of rows: option E's group heading (17 Display, sentence case). `step` is ignored (E has no
// numbered steps); `trailW`: room kept at the right end. Returns its height (22).
float SectionLabel(Gfx& gx, float x, float y, float w, const wstring& text, int step = 0, float trailW = 0);
void Bevel(Gfx& gx, const Rc& r, float rad, Col fill, Col edge);  // a flat fill with a 1 px edge (the "Bench" name)
void Led(Gfx& gx, float cx, float cy, Col c, bool lit = true, float r = 4);  // = StatusDot (r and lit kept for old callers)
Col StatusLed(const GameEntry& e, bool& lit);           // green running fine, amber needs a look, grey off
void Brushed(Gfx& gx, const Rc& r);                     // nothing in option E (kept for old callers)
wstring When(const wstring& when);                       // "2026-10-17 19:19" -> "Today at 19:19"
void NameColours(const wstring& name, Col& a, Col& b);  // an accent and a dark partner from the name (generated art)
void GeneratedArt(Gfx& gx, const Rc& r, const wstring& name, bool bar = true);  // the art of a game without Steam art (a graphite plate, the name's accent)
wstring ProblemShort(const GameEntry& e);               // "A WrongGPU file was changed" (tiles)
// The game a converted model file came from ("Forza Horizon 6"), or empty. Names come from `libs` when given (worker
// threads), else from the Library (UI thread only).
wstring ModelSourceGame(const wstring& sourcePath, const std::vector<StoreGame>* libs = nullptr);
wstring ProblemSentence(const wstring& problem);        // "wronggpu_runtime.dll was changed after the install." (the game page)
wstring FitMiddle(Gfx& gx, const wstring& t, float width, const TS& s);  // a path shortened in the middle with an ellipsis
// A folder in words for a sentence, its whole path left for a tooltip or Show details ("Itch in K: › Games", "Downloads in
// your user folder"; ui_settings.cpp's WordsFor)
wstring PlaceWords(const wstring& path);
// A sentence whose last line never holds one short word alone ("need it."): its last two spaces become no-break spaces
// when the words after them are short
wstring NoWidow(const wstring& t);
wstring StoreOf(const GameEntry& e);                   // Steam, Epic, GOG, ...
float Callout(Gfx& gx, int id, float x, float y, float w, const wstring& text, int tone, const wstring& link = L"", wchar_t icon = 0);
// A text field (search, the release address, OptiScaler numbers) on the fill, radius 9; the native EDIT sits on it.
// clearId: a clear button (x) at the right end while the field holds text. enabled: false draws it dimmed and takes no
// click, focus or keys (a locked row)
void FieldBox(Gfx& gx, EditBox& e, int id, const Rc& r, const wstring& text, const wstring& cue, wchar_t icon, const wstring& name, bool mono = false,
              int clearId = 0, bool enabled = true);
// An EDIT child is shown when its whole field is in the part of the window its page shows content in (not under the game
// page's pinned plate or its unsaved bar, not cut off at an edge), the page is not making its entrance, no dialog or first
// run is open and no open menu covers it. Otherwise FieldBox draws the field's text itself.
bool EditShown(const EditBox& e);
// the mark: a black rounded square, a white "W" out of register (red / green), a red bar; fill: the bar's green part from the
// left, cross: W -> R (RightGPU)
void AppMark(Gfx& gx, const Rc& r, float fill = 0, float cross = 0);
bool RightGpu(const GameEntry& e);   // WrongGPU installed and healthy in this game: its page shows RightGPU
bool BrandNow(float& fill, float& cross);  // the mark's state now (fill, cross 0..1); true while the flip still moves
void DrawMenu(Gfx& gx);
Rc MenuRect();                   // where the open menu is drawn (window DIPs)
float MenuItemH();
float MenuMaxScroll();           // a menu longer than its room scrolls (0: it fits)
bool MenuScroll(float dy);       // the wheel over a menu that scrolls; false: it fits (nothing to scroll)
void MenuShowItem(int index);    // the keyboard's item brought into the menu's view
void DrawDialog(Gfx& gx);
// The button Return answers (drawn filled): a note's one button; for a question about an action the player chose, the
// action; -1 for a warning or a destructive action (its buttons are drawn as equals; RunDialog puts the keyboard focus on
// Cancel, so Return cancels, as Windows and macOS have it)
int DialogDefault(const Dialog& d);
void DrawToast(Gfx& gx);
void DrawScrollbar(Gfx& gx, Page p);
float ThumbAlpha();              // how much the overlay scroll thumb shows now (0 hidden .. 1)
void DrawCaption(Gfx& gx);       // the caption buttons (Minimize, Maximize / Restore, Close) at the toolbar's right end
void OpenMenu(int owner, const Rc& anchor, std::vector<MenuItem> items, bool right, float width = 220);
std::vector<MenuItem> PageMoreMenu();  // the game page's More menu as it is now (MoreMenu, OptiScaler's reset among its items)
void RefreshPageMenu();                // the page read again for its game: an open More / Output filter menu keeps up (else closes)
void ShowToast(const wstring& text, int tone);
// Motion (entrances of the page, menus, dialogs, toasts and the first-run card): how far an entrance that started at `at`
// is, eased (0..1). Always 1 offscreen and when Windows' "Animation effects" are off (SPI_GETCLIENTAREAANIMATION).
bool MotionOn();
void MotionRefresh();  // Windows' settings changed: Animation effects, Contrast themes and Transparency read again when next asked
// Milliseconds on GetTickCount64's scale, but fine-grained (QueryPerformanceCounter): entrances move on every frame at
// 144 Hz instead of in 15.6 ms steps. Stamps taken with either clock can be compared.
ULONGLONG NowMs();
float Entrance(ULONGLONG at, float ms);
bool MotionPending();  // an entrance or a toast's fade is still moving (the window keeps its animation timer)
bool PageEntering();   // the page is making its entrance (the native text fields wait, drawn by FieldBox until it ends)
// A dialog or the first-run sheet is open and has made its entrance: the window behind it shows as the scrim's frozen
// picture (Scrim, Gfx::DrawFrozen), so the page's own motion under it (a progress bar's light, the scan's skeleton) never
// reaches the screen and asks for no frames
bool PageUnderSheet();
constexpr float kPageMs = 180, kMenuMs = 140, kDialogMs = 180, kToastMs = 220, kToastOutMs = 260, kFirstMs = 240;
constexpr float kThumbHoldMs = 900, kThumbFadeMs = 300;  // the overlay scroll thumb: shown this long after the page last moved, then fades
// Icons (Segoe Fluent Icons / Segoe MDL2 Assets code points)
namespace ic {  // one per line
const wchar_t grid = 0xF0E2;
const wchar_t gear = 0xE713;
const wchar_t refresh = 0xE72C;
const wchar_t add = 0xE710;
const wchar_t download = 0xE896;
const wchar_t search = 0xE721;
const wchar_t back = 0xE76B;
const wchar_t play = 0xF5B0;
const wchar_t more = 0xE712;
const wchar_t warn = 0xE7BA;
const wchar_t folder = 0xE8B7;
const wchar_t cube = 0xE7B8;
const wchar_t panel = 0xEA86;
const wchar_t monitor = 0xE7F4;
const wchar_t down = 0xE70D;
const wchar_t check = 0xE73E;
const wchar_t close = 0xE711;
const wchar_t info = 0xE946;
const wchar_t doc = 0xE8A5;
const wchar_t zip = 0xF012;
const wchar_t wrench = 0xE90F;
const wchar_t undo = 0xE7A7;
const wchar_t trash = 0xE74D;
const wchar_t broom = 0xEA99;
const wchar_t upload = 0xE898;
const wchar_t link = 0xE71B;
const wchar_t minus = 0xE738;
const wchar_t shield = 0xEA18;
const wchar_t question = 0xE897;
const wchar_t chip = 0xE964;
const wchar_t sync = 0xE895;
const wchar_t key = 0xE8D7;
const wchar_t playLine = 0xE768;
const wchar_t up = 0xE70E;
const wchar_t arrowUp = 0xE74A;  // an update ready (UpdateMark)
const wchar_t palette = 0xE790;
const wchar_t chevron = 0xE76C;
const wchar_t external = 0xE8A7;
const wchar_t processor = 0xE950;
}

// ---- pages (ui_library.cpp, ui_game.cpp, ui_settings.cpp) ---------------------------------------------------------------
void PaintLibrary(Gfx& gx, const Rc& area);
bool LibraryMoving();             // the Library's last frame moved (a cover's fade, a progress bar's travel, the scan's step)
// The Library as the last session left it (library-list.txt in the launcher's data folder): read at start-up and shown
// at once while the scan reads the games again (true: v.games set, v.listCached), written when a scan lands and when
// the window closes. Live windows only.
bool LoadLibrarySnapshot();
void SaveLibrarySnapshot();
void PaintGame(Gfx& gx, const Rc& area);
void PaintGameOverlay(Gfx& gx, const Rc& area);  // the "unsaved changes" bar docked at the bottom
float GameBarH();                 // that bar's height (0: settings match the profile, or another page)
bool GameMotion();                // the game page's own motion (the unsaved capsule coming or going): MotionPending
void PaintSettings(Gfx& gx, const Rc& area);
float SettingsSectionScroll(int section);  // the scroll that brings a Settings section's heading to the top (its index link)
void PaintFirstRun(Gfx& gx);
// The words of a model conversion that worked, for its toast (UI thread: the source game as the Library names it):
// "Model M data is converted again from Cyberpunk 2077's DLSS file." `again`: a model was replaced; `chose`: the player picked the file.
wstring ModelConvertedWords(bool again, bool chose);
wstring HipRuntimeWords(const wstring& dll);  // "C:\...\amdhip64_7.dll" -> "HIP 7" (empty when the name says no version)
// The arrow keys on the Library's lockups and its shelf buttons: the id to focus, -1 with nothing that way (Up and Down
// then scroll), 0 not on a lockup.
int LibraryArrow(int focus, WPARAM key);
int NeedsAttentionCount();
// ui_library.cpp: a protected game's Engine.ini line ready to add on the player's click ("Needs attention", Fix it now; r3 #5)
bool RrConfigReadyToAdd(const GameEntry& e);
wstring LockupSaid(const GameEntry& e);  // the Library tile's state for UI Automation (LockupStatus; the ui-dump, r7 #6)
float LibraryBannerH(float viewH);  // ui_library.cpp: the featured banner's height in a view this high
int ProblemCount();  // ui_library.cpp: the attention games that are not just an update (or an Engine.ini line ready to add: r4 #8)
bool RedAttention(const GameEntry& e);  // ui_library.cpp: this game counts in ProblemCount (the tab's count red)
bool GuideShown();  // ui_game.cpp: the game page shows "Turn it on in the game" (UI review #1)
bool RrHeldOffHere();       // ui_game.cpp (rc5): the page's game takes RR only through its experimental opt-in, now off
bool RrExperimentalHere();  // ui_game.cpp (rc5): ... and that opt-in is on (the RR switch turns it off)
bool AdvancedWanted();  // ui_game.cpp: a page opens with its Advanced rows shown (a value off its default, an outside edit, an unsaved one)
bool LockupPlays(const GameEntry& e);  // an installed game's lockup says Play (else Details); Activate does the same
int UpdateCount();
wstring ShortStatus(const GameEntry& e);
wstring ExecutorName(const wstring& e);  // dx12 -> D3D12
// Ray Reconstruction (rr.h): the game's install has WrongGPU's RR files and RR on (the status lines then say "Model M and
// Ray Reconstruction"); a mode in the page's words ("accurate" -> "Accurate")
bool RrOn(const GameEntry& e);
// ...and its last run was Ray Reconstruction alone (rrOnly) or RR did the upscaling in it (rrUpscaled): Model M had no part in it,
// so the page's header and the Library's banner say "Ray Reconstruction", not "Model M and Ray Reconstruction" (r5 #4, r6 #1, r6 #6)
bool RrAloneLastRun(const GameEntry& e);
wstring RrModeName(const wstring& mode);
bool Dirty();
bool PageWritesWait();  // the page's values can't be written now (an operation, the last session's list, the game running)
bool PendingEdits();    // the write under way (v.pendingApply) carries changes made on the page (not only a callout's button)
void ShowReadingPage(const GameEntry& e, int index);  // a game's page before its files are read (OpenGame, View::pageReading)
void OpenGameFolder(const GameEntry& e);  // More > Open folder (a folder that is gone: said in words)
void OpenGameLog(const GameEntry& e);     // More > Open log (none yet, or the folder gone: said in words); both off the window's thread
void OffThread(std::function<void()> f);  // launcher_ui.cpp: `f` on a worker (counted: the window waits for it before the process ends)
void NoteFrom(HWND hwnd, const wstring& text, int icon);  // launcher_ui.cpp: a note from a worker, shown once no dialog is open
wstring BusyDoneWords();  // what the running operation waits on, for "... once <this>.": "the update is done", "Quarry Nine's repair is done"
std::vector<MenuItem> MoreMenu();
std::vector<MenuItem> FilterMenu();     // the output filter's choices (each with its line)
wstring KeyName(const string& value);   // "0x2D" -> "Insert"
extern const int kMenuKeys[20];         // the keys the "Menu key" select offers
inline Context& Ctx() { return wg::app::g.ctx; }
bool ThemeIsDark(const wstring& appearance);  // "system": Windows' app mode (AppsUseLightTheme)

// ---- the window's actions (launcher_ui.cpp) --------------------------------------------------------------------------
void Activate(int id);           // a click, Enter or Space on a widget
void MenuCommand(int owner, int id);
void Invalidate();
void FocusId(int id);            // the keyboard focus to a widget, as Tab moves it (into view, a text field's keys, UI Automation's event)
void RevealId(int id);           // a widget or words brought into view without the focus (UI Automation's ScrollIntoView)
// danger: the action destroys something (Uninstall, Clean up, Reset, Remove): its button is red and never the default
bool Confirm(const wstring& text, int icon = 0, const wstring& yes = L"Yes", const wstring& no = L"No", bool danger = false);
void Inform(const wstring& text, int icon = 2);
// A note that arrives on its own (a worker's result): shown at once, or once the dialog or file dialog that is open closes, so
// it never answers a question the player has open.
void InformLater(const wstring& text, int icon = 2);
void OpenGame(int index);
void PaintAll(Gfx& gx);
const std::vector<Hit>& CurrentHits();
wstring BuildLine();             // "WrongGPU 0.5.5 \u00b7 portable"

// ---- UI Automation (ui_uia.cpp) -----------------------------------------------------------------------------------------
LRESULT UiaGetObject(HWND hwnd, WPARAM wp, LPARAM lp);
void UiaFocusChanged(int id);
// An announcement (Narrator speaks it): a toast, a dialog's question, the unsaved capsule, an operation that starts.
// important: spoken at once (else after what it is saying). kind: 0 a state or an event, 1 an action's result (it worked),
// 2 an action that failed (UI Automation's notification kinds: ActionCompleted, ActionAborted).
void UiaNotify(const wstring& text, bool important = false, int kind = 0);
// After every frame of the live window: state changes of the frame's elements reported (a switch toggled, a segment or a
// menu item chosen, a select's value, a fold opened, a menu or dialog opened, the page's elements replaced).
void UiaFrame();
HWND EditOf(int id);  // the native EDIT of a text field's hit (search, release address, an OptiScaler number), else null
void UiaNameEdit(HWND h, const wstring& name);  // the native EDIT's accessible name (Narrator reads it when it has the keys)
void UiaForgetEdit(HWND h);                     // its name dropped (the EDIT is being destroyed)
void UiaMarkEdit(HWND h, const wstring& why);   // the EDIT's value refused (why: its help text) or, empty, valid again
wstring FieldInvalidWhy(int id);                // a text field's hit whose value Apply refused: why (v.optiBad), else empty
void UiaShutdown(HWND hwnd);
bool UiaInvokeMessage(UINT msg, WPARAM wp, LPARAM lp);  // Invoke / Toggle / Scroll from UI Automation, run on the window's thread

}  // namespace wg::app::lui
