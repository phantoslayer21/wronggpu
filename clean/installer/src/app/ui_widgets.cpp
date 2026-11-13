// SPDX-License-Identifier: MIT
// WrongGPU launcher window: the theme tokens and the widgets (ui_shell.h) in option E, "Quiet": capsule buttons, pills,
// switches, segmented controls, filter chips, selects, text fields, inset grouped lists with hairlines, covers, the
// toolbar (the mark, Library | Settings, the status line), menus, dialogs, toasts, the floating capsule's blurred
// material, the scrim and the scrollbar. Each widget draws itself for the current frame and adds a hit (ui_shell.h Hit)
// for the mouse, the keyboard, UI Automation and the ui-dump. Sizes and colours are the mockup's
// (build/launcher/ui-options/E-quiet/index.html, light and dark). Written for this project.
#include "app/ui_shell.h"

namespace wg::app::win { extern ULONGLONG clickGuardAt; }  // launcher_ui.cpp: a double click's second click
namespace wg::app::win { wstring RrBackgroundStatus(); }  // ui_rrprep.cpp: RR prepared again after a driver update (0.6.0 polish)

namespace wg::app::lui {

View v;
ContentView cv;
Theme pal;

// ---- theme ------------------------------------------------------------------------------------------------------------------
static int themeAccent = A_GREEN;
static unsigned themeWindowsRgb = 0x0078d4;
void SetAccent(int accent, unsigned windowsRgb) { themeAccent = accent >= 0 && accent < A_COUNT ? accent : A_GREEN; themeWindowsRgb = windowsRgb & 0xffffff; }
int CurrentAccent() { return themeAccent; }
// Windows' accent colour: DWM's AccentColor (0xAABBGGRR, what Settings > Personalization > Colors picks), else its
// colorization colour (0xAARRGGBB); Windows' default blue when neither is there
unsigned WindowsAccentRgb() {
    DWORD c = 0, size = sizeof c;
    if (RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\DWM", L"AccentColor", RRF_RT_REG_DWORD, nullptr, &c, &size) == ERROR_SUCCESS)
        return ((c & 0xff) << 16) | (c & 0xff00) | ((c >> 16) & 0xff);
    size = sizeof c;
    if (RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\DWM", L"ColorizationColor", RRF_RT_REG_DWORD, nullptr, &c, &size) == ERROR_SUCCESS)
        return c & 0xffffff;
    return 0x0078d4;
}
// Windows' Contrast themes (SPI_GETHIGHCONTRAST) and Transparency effects (Personalize\EnableTransparency; missing: on): read
// once and kept, read again after WM_SETTINGCHANGE (MotionRefresh). ui-render can draw a contrast theme's colours without
// the PC using one (ContrastPreview: 1 Night sky, 2 Desert; their system colours as Windows 11 ships them).
static int contrastOn = -1, transparencyOn = -1, contrastPreview = 0;
void ContrastPreview(int preset) { contrastPreview = preset; }
bool HighContrast() {
    if (contrastPreview) return true;
    if (contrastOn < 0) {
        HIGHCONTRASTW hc{sizeof hc};
        contrastOn = SystemParametersInfoW(SPI_GETHIGHCONTRAST, sizeof hc, &hc, 0) && (hc.dwFlags & HCF_HIGHCONTRASTON) ? 1 : 0;
    }
    return contrastOn != 0;
}
bool TransparencyOn() {
    if (transparencyOn < 0) {
        DWORD value = 1, size = sizeof value;
        if (RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", L"EnableTransparency", RRF_RT_REG_DWORD, nullptr,
                         &value, &size) != ERROR_SUCCESS)
            value = 1;
        transparencyOn = value != 0;
    }
    return transparencyOn != 0;
}
static Col SysCol(int index) {
    if (contrastPreview) {
        // window, window text, gray text, highlight, highlight text, hot light
        static const unsigned night[] = {0x000000, 0xffffff, 0xa6a6a6,  // Night sky
                                         0xd6b4fd, 0x2b2b2b, 0x8080ff};
        static const unsigned desert[] = {0xfffaef, 0x3d3d3d, 0x676767,  // Desert
                                          0x903909, 0xfff5e3, 0x1c5e75};
        const unsigned* p = contrastPreview == 2 ? desert : night;
        const int i = index == COLOR_WINDOW ? 0 : index == COLOR_WINDOWTEXT ? 1 : index == COLOR_GRAYTEXT ? 2 : index == COLOR_HIGHLIGHT ? 3 : index == COLOR_HIGHLIGHTTEXT ? 4 : 5;
        return Hex(p[i]);
    }
    const DWORD c = GetSysColor(index);
    return Col{GetRValue(c) / 255.f, GetGValue(c) / 255.f, GetBValue(c) / 255.f, 1};
}
void SetTheme(bool dark) {
    Theme t;
    t.dark = dark;
    t.black = Hex(0x000000); t.white = Hex(0xffffff);
    t.onAccent = Hex(0xffffff); t.knob = Hex(0xffffff);
    const Surfaces sf = SurfacesOf(dark);
    t.bg = sf.bg; t.cell = sf.cell; t.fill = sf.fill;
    if (!dark) {
        // light: its own surfaces (a soft grey page, white cells with a faint shadow), not an inversion of the dark theme
        t.sep = Hex(0xe6e6ea);
        t.text = Hex(0x1d1d1f); t.text2 = Hex(0x606067); t.text3 = Hex(0x85858c);  // text3: 3.7:1 on a cell, 3.3 on the page, 3.1 on the fill
        t.fillHover = Hex(0xe2e2e7); t.segOn = Hex(0xffffff);
        t.accent = Hex(0xc42b1c); t.accentText = Hex(0xb8261a); t.accentTint = Hex(0xfae9e7); t.accentTintHover = Hex(0xf6dcd9);
        t.regR = Hex(0xe23028, 0.55f); t.regG = Hex(0x1ea046, 0.55f);
        // the dots (graphics, WCAG 1.4.11): amber 3.6:1 on a cell, 3.3 on the page, 3.1 on the fill; grey the same
        t.amber = Hex(0xbf7600); t.amberText = Hex(0x94600a); t.greyDot = Hex(0x86868d);
        t.trackOff = Hex(0xc4c4ca); t.trackEdge = Hex(0x8a8a91);  // the edge: 3.4:1 on a cell (the fill alone is 1.7:1)
        t.sheet = Hex(0xffffff); t.material = Hex(0xffffff, 0.82f); t.scrim = Hex(0x1e1e24, 0.28f);
        t.focus = Hex(0x1d1d1f);
        t.cellShadowA = 0.05f; t.shadowA = 0.18f;
    } else {
        // dark: true black (the unlock), cells and fills a step up from it
        t.sep = Hex(0x2a2a2e);
        t.text = Hex(0xf2f2f4); t.text2 = Hex(0xa4a4ab); t.text3 = Hex(0x7e7e85);  // text3: 4.5:1 on a cell, 5.2 on the page, 3.9 on the fill
        t.fillHover = Hex(0x2c2c30); t.segOn = Hex(0x3a3a3f);
        t.accent = Hex(0xc9362a); t.accentText = Hex(0xff8579); t.accentTint = Hex(0x3b2523); t.accentTintHover = Hex(0x4a2c29);
        t.regR = Hex(0xff5046, 0.55f); t.regG = Hex(0x50dc78, 0.55f);
        t.amber = Hex(0xf0a326); t.amberText = Hex(0xf0b44c); t.greyDot = Hex(0x76767e);
        t.trackOff = Hex(0x3a3a3f); t.trackEdge = Hex(0x74747b);  // the edge: 3.9:1 on a cell (the fill alone is 1.6:1)
        t.sheet = Hex(0x1c1c1e); t.material = Hex(0x1c1c1e, 0.86f); t.scrim = Hex(0x000000, 0.5f);
        t.focus = Hex(0xf2f2f4);
        t.cellShadowA = 0; t.shadowA = 0.5f;
    }
    // green: WrongGPU is working (fixed); the accent: the player's pick (measured in ui_palette.h's tests)
    t.green = StatusGreen(dark); t.greenText = StatusGreenText(dark);
    const AccentTokens k = AccentFor(themeAccent, dark, themeWindowsRgb);
    t.tint = k.c; t.tintFill = k.f; t.tintText = k.t; t.tintBg = k.tint; t.tintBgHover = k.tintHover;
    t.edge = t.sep;
    if (HighContrast()) {
        // a Contrast theme (Windows' Accessibility > Contrast themes, WinUI's and Apple's Increase Contrast): the system's own
        // colours, as the player chose them, for every surface, word and line (COLOR_WINDOW, COLOR_WINDOWTEXT, COLOR_GRAYTEXT,
        // COLOR_HIGHLIGHT and its text, COLOR_HOTLIGHT for links); no translucency, no shadows, no registration offset; the
        // widgets draw an edge round every control (hc) instead of telling them apart by fills
        const Col win = SysCol(COLOR_WINDOW), ink = SysCol(COLOR_WINDOWTEXT), gray = SysCol(COLOR_GRAYTEXT), hl = SysCol(COLOR_HIGHLIGHT),
                  hlInk = SysCol(COLOR_HIGHLIGHTTEXT), link = SysCol(COLOR_HOTLIGHT);
        t.hc = true; t.dark = Luminance(win) < 0.5f;
        t.bg = t.cell = t.fill = t.fillHover = t.segOn = t.sheet = t.material = win;
        t.sep = t.edge = ink;
        t.text = t.text2 = ink; t.text3 = gray;
        t.accent = hl; t.onAccent = hlInk; t.accentText = ink; t.accentTint = t.accentTintHover = win;
        t.tint = t.tintFill = hl; t.tintText = link; t.tintBg = t.tintBgHover = win;
        t.green = hl; t.greenText = ink; t.amber = ink; t.amberText = ink; t.greyDot = gray;
        t.trackOff = win; t.trackEdge = ink; t.knob = ink;
        t.regR = Alpha(ink, 0); t.regG = Alpha(ink, 0);
        t.scrim = Alpha(win, 0.6f);  // drawn at 1.5 times without the blur (Scrim): the page behind washed toward the window colour
        t.focus = ink;
        t.cellShadowA = 0; t.shadowA = 0;
    }
    // the "Bench" names on the E tokens (the pages still using them look like E already)
    t.alu = t.bg; t.panel = t.cell; t.ink = t.text; t.graphite = t.text2; t.faint = t.text3; t.groove = t.sep;
    t.red = t.accent; t.redEdge = Mix(t.accent, t.black, 0.15f); t.onRed = t.onAccent; t.redText = t.accentText; t.off = t.greyDot;
    t.hl = Hex(0xffffff, 0); t.under = Hex(0x000000, t.cellShadowA); t.sel = t.segOn; t.selInk = t.text;
    t.rk = t.trackOff; t.knobTop = t.knob; t.knobBottom = t.knob;
    t.hover = t.fill; t.press = t.fillHover; t.well = t.fill; t.bezel = t.bg;
    t.brushLight = Hex(0xffffff, 0); t.brushDark = Hex(0x000000, 0); t.ledRing = Hex(0x000000, 0);
    t.onArt = Hex(0xffffff); t.onArtDim = Hex(0xd5d9df); t.artShade = Hex(0x000000);
    t.artTint = Hex(0x3b2523); t.artTintHover = Hex(0x4a2c29); t.artTintText = Hex(0xff8579);  // the dark tint pill: its words 6.0:1, both themes
    t.genA = Hex(0x343a42); t.genB = Hex(0x1d2126); t.genInk = Hex(0xe9ecef);
    // on the art's scrim (about #434100 to #4a4a1a on Cyberpunk's yellow, darker elsewhere): the ring 8:1, green 5.7:1, amber
    // 4.8:1, grey 4.2:1, red 4.0:1 at the brightest
    t.focusArt = Hex(0xf2f2f4); t.greenArt = StatusGreen(true); t.amberArt = Hex(0xf0a326); t.greyArt = Hex(0xa4a4ab); t.redArt = Hex(0xff8579);
    pal = t;
}

// Windows' app mode: HKCU\...\Themes\Personalize AppsUseLightTheme (0: dark). Missing: light.
bool ThemeIsDark(const wstring& appearance) {
    if (appearance == L"dark") return true;
    if (appearance == L"light") return false;
    DWORD value = 1, size = sizeof value;
    if (RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", L"AppsUseLightTheme", RRF_RT_REG_DWORD, nullptr,
                     &value, &size) != ERROR_SUCCESS)
        return false;
    return value == 0;
}

// ---- hits -------------------------------------------------------------------------------------------------------------------
void AddHit(int id, Rc r, Role role, const wstring& name, bool enabled, int state, const wstring& value, bool focusable) {
    if (cv.dry) return;
    const Rc full = r.off(0, -cv.scroll);
    Rc w = full;
    if (cv.view.w > 0) w = Intersect(w, cv.view);
    bool shown = true;
    if (w.w <= 0.5f || w.h <= 0.5f) {  // scrolled out of view: kept (unclipped, not shown) for Tab and UI Automation
        if (cv.view.w <= 0 || full.w <= 0.5f || full.h <= 0.5f || full.r() <= cv.view.x || full.x >= cv.view.r()) return;
        w = full; shown = false;
    }
    Hit h; h.id = id; h.r = w; h.role = role; h.shown = shown; h.name = name; h.enabled = enabled; h.state = state; h.value = value; h.focusable = focusable && enabled;
    h.full = full; h.scroll = cv.scroll;
    v.hits.push_back(h);
}
// The frame's next free id for words (ID_AUTO0 + n): one more than the highest so far (the hits are few hundred at most)
static int NextAutoId() {
    int next = ID_AUTO0;
    for (auto& h : v.hits) if (h.id >= next && h.id < ID_AUTO0 + kAutoSpan) next = h.id + 1;
    return std::min(next, ID_AUTO0 + kAutoSpan - 1);
}
int AddText(const Rc& r, const wstring& text, int id, int heading) {
    if (cv.dry || text.empty()) return 0;
    if (!id) id = NextAutoId();
    AddHit(id, r, R_TEXT, text, true, heading > 0 ? heading : -1, L"", false);
    return id;
}
void AddProgress(int id, const Rc& r, const wstring& name, float fraction) {
    if (cv.dry) return;
    if (!id) id = NextAutoId();
    const int pct = fraction < 0 ? -1 : (int)std::lround(std::min(1.f, std::max(0.f, fraction)) * 100);
    AddHit(id, Rc{r.x, r.y - 6, r.w, std::max(r.h, 16.f)}, R_PROGRESS, name, true, pct, L"", false);  // a 4 DIP bar: a box UI Automation can point at
}
void Describe(int id, const wstring& help) {
    if (cv.dry || help.empty()) return;
    for (size_t i = v.hits.size(); i-- > 0;) if (v.hits[i].id == id) { v.hits[i].help = help; return; }
}
float Measure(Gfx& gx, const std::function<float()>& paint) {
    const bool d0 = gx.dry, d1 = cv.dry;
    gx.dry = cv.dry = true;
    const float h = paint();
    gx.dry = d0; cv.dry = d1;
    return h;
}
bool Hot(int id) { return v.hot == id; }
bool Down(int id) { return v.down == id && v.hot == id; }
bool Focused(int id) { return v.focusVisible && v.focus == id; }
// The keyboard focus: a 2 px ring in the text colour, 2 DIPs outside the part (the mockup's outline-offset), following its
// corners.
static int onArtDepth = 0;
ArtScope::ArtScope() { ++onArtDepth; }
ArtScope::~ArtScope() { --onArtDepth; }
// the focus ring last drawn (window DIPs, its colour), for FocusRingOverBar
static struct { int id = 0; Rc r; float rad = 0; Col c; } lastRing;
void FocusRing(Gfx& gx, const Rc& r, float rad, int id) {
    if (!Focused(id)) return;
    const Col c = onArtDepth > 0 ? pal.focusArt : pal.focus;
    gx.StrokeRound(r.in(-4), rad + 4, c, 2);
    if (!gx.dry) { lastRing.id = id; lastRing.r = r.off(0, -cv.scroll); lastRing.rad = rad; lastRing.c = c; }
}
void FocusRingOverBar(Gfx& gx) {
    const auto ring = lastRing;
    lastRing.id = 0;
    if (!ring.id || ring.id != v.focus || v.dialog.open || v.first.open || PageEntering()) return;
    const Rc rr = ring.r.in(-4);
    // only a control wholly under the toolbar whose ring reaches over its edge (one scrolled under it stays cut off)
    if (rr.y - 1 >= kBarH || ring.r.y < kBarH - 0.5f) return;
    gx.PushClip(Rc{0, kBarH - 8, v.W, 8});
    gx.StrokeRound(rr, ring.rad + 4, ring.c, 2);
    gx.PopClip();
}

static TS Ts(float size, int weight, Col c, FontFace f = F_TEXT) { TS s; s.size = size; s.weight = weight; s.color = c; s.face = f; return s; }

// ---- surfaces ---------------------------------------------------------------------------------------------------------------
void Brushed(Gfx&, const Rc&) {}
void Bevel(Gfx& gx, const Rc& r, float rad, Col fill, Col edge) {
    gx.FillRound(r, rad, fill);
    gx.StrokeRound(r, rad, edge);
}
// A Contrast theme's edge round a control (Theme::hc): the window text colour, the highlight under the pointer; nothing
// otherwise (the fills tell controls apart)
void HcEdge(Gfx& gx, const Rc& r, float rad, bool hot, bool enabled) {
    if (!pal.hc) return;
    gx.StrokeRound(r.in(hot ? 1.f : 0.5f), std::max(0.f, rad - (hot ? 1.f : 0.5f)), !enabled ? pal.text3 : hot ? pal.tint : pal.edge, hot ? 2.f : 1.f);
}
// A rectangle's edges on whole device pixels (content coordinates: the scroll and an entrance move by whole pixels): a list's
// cell at 150 % has hard edges, as its hairlines (Hairline snaps the same way), not a half-tone row at its top and bottom
static Rc SnapPx(const Gfx& gx, const Rc& r) {
    const float s = std::max(1.f, gx.scale);
    const float x0 = std::round(r.x * s) / s, y0 = std::round(r.y * s) / s;
    return Rc{x0, y0, std::round(r.r() * s) / s - x0, std::round(r.b() * s) / s - y0};
}
void GroupBox(Gfx& gx, const Rc& rIn, float rad) {
    const Rc r = SnapPx(gx, rIn);
    if (pal.cellShadowA > 0) gx.FillRound(r.off(0, 1), rad, Alpha(pal.black, pal.cellShadowA));  // 0 1px 2px rgba(0,0,0,.05)
    gx.FillRound(r, rad, pal.cell);
    HcEdge(gx, r, rad);
}
void Card(Gfx& gx, const Rc& r, float rad) { GroupBox(gx, r, rad); }

void Sheet(Gfx& gx, const Rc& r, float rad, float lift) {
    // 0 10px 30px rgba(0,0,0,.22) (menus); a sheet 0 24px 70px rgba(0,0,0,.30) (lift 2.3)
    if (!pal.hc) gx.Shadow(r, rad, 30 * lift, pal.dark ? 0.6f : lift > 1.5f ? 0.41f : 0.3f, 10 * lift, true);  // the sheet's fill is opaque
    gx.FillRound(r, rad, pal.sheet);
    gx.StrokeRound(r, rad, pal.sep, pal.hc ? 2.f : 1.f);               // 0 0 0 1px var(--sep)
}

void Material(Gfx& gx, const Rc& r, float rad, float blur, float opacity) {
    if (pal.shadowA > 0) gx.Shadow(r, rad, 28, pal.shadowA * 1.3f * opacity, 8);  // 0 8px 28px rgba(0,0,0,.18)
    // Transparency effects off, or a Contrast theme: the opaque sheet, nothing of the page through it
    const bool blurred = !pal.hc && TransparencyOn() && gx.Backdrop(r, rad, blur, opacity);
    gx.FillRound(r, rad, blurred ? Alpha(pal.material, opacity) : Alpha(pal.sheet, 0.98f * opacity));
    gx.StrokeRound(r, rad, Alpha(pal.sep, opacity));
}

// What the window behind a sheet shows, as one number: while it stays the same, the blurred and dimmed picture made once the
// sheet's entrance ended is drawn again (Gfx::Freeze) instead of copying and blurring the whole window on every frame (a
// pointer over a dialog's buttons, a first-run line). A scan's result, a toast, a theme or size change make it again.
static unsigned long long BackdropKey() {
    unsigned long long h = 1469598103934665603ULL;
    auto mix = [&](unsigned long long x) { h = (h ^ x) * 1099511628211ULL; };
    auto str = [&](const wstring& s) { mix(std::hash<wstring>()(s)); };
    mix((unsigned long long)std::lround(v.W * 4)); mix((unsigned long long)std::lround(v.H * 4)); mix(pal.dark); mix(pal.hc); mix(TransparencyOn());
    mix((unsigned long long)CurrentAccent());
    mix((unsigned long long)v.page); mix((unsigned long long)std::lround(v.scroll[v.page])); mix(v.games.size()); mix(v.scanning); mix(v.busy); mix((unsigned long long)v.task);
    mix((unsigned long long)std::lround(v.progress * 100)); str(v.step); str(v.busyDir); mix(v.modelReady); str(v.executorLine); str(v.cur.dir); str(v.notice);
    mix(v.attnAll); str(v.search); mix(v.menu.open); mix(v.queueDone);
    mix(v.first.open); mix(v.first.busy); mix(v.first.lines.size()); mix((unsigned long long)v.first.choice); mix(v.first.checksOpen);
    mix((unsigned long long)std::lround(v.first.scroll));
    const bool toast = !v.toast.text.empty() && NowMs() <= v.toast.until;
    mix(toast); if (toast) str(v.toast.text);
    for (auto& ge : v.games) mix((unsigned long long)ge.status);
    return h | 1;
}
void Scrim(Gfx& gx, float amount) {
    if (amount <= 0) return;
    const Rc all{0, 0, v.W, v.H};
    const unsigned long long key = amount >= 1 && v.live ? BackdropKey() : 0;
    if (key && gx.DrawFrozen(key)) return;
    const bool blurred = !pal.hc && TransparencyOn() && gx.Backdrop(all, 0, 6 * amount, 1);
    // without the blur the page shows through sharp: a little more dim keeps the sheet in front
    gx.Fill(all, Alpha(pal.scrim, amount * (blurred ? 1.f : 1.5f)));
    if (key) gx.Freeze(key);
}

void BasePlate(Gfx& gx, const Rc& sheet, float h, float) { Hairline(gx, sheet.x + 1, sheet.b() - h, sheet.r() - 1); }

Col PageText() { return pal.text2; }

float GroupHeading(Gfx& gx, float x, float y, float w, const wstring& title, const wstring& count, float trailW) {
    const TS ts = Ts(17, 600, pal.text, F_DISPLAY);
    const float room = std::max(0.f, w - 8 - trailW);
    const float tw = std::min(room, gx.TextW(title, ts) + 1);
    gx.Text(title, Rc{x + 4, y, tw + 1, 23}, ts);
    if (!count.empty() && tw + 6 < room) gx.Text(count, Rc{x + 4 + tw + 6, y, room - tw - 6, 23}, Ts(17, 400, pal.text2, F_DISPLAY));
    AddText(Rc{x + 4, y, tw + 1, 23}, count.empty() ? title : title + L", " + count, 0, 2);  // a heading (Narrator's heading keys)
    return 22;
}
float GroupCaption(Gfx& gx, float x, float y, float w, const wstring& text) {
    TS ts = Ts(13, 400, pal.text2); ts.wrap = true; ts.lineH = 18;
    const float th = gx.TextH(text, w - 8, ts);
    gx.Text(text, Rc{x + 4, y - 6, w - 8, th + 2}, ts);  // pulled up under its heading (margin -6px 4px 10px)
    AddText(Rc{x + 4, y - 6, w - 8, th + 2}, text);
    return th - 6 + 10;
}
float GroupFoot(Gfx& gx, float x, float y, float w, const wstring& text) {
    TS ts = Ts(12.5f, 400, pal.text2); ts.wrap = true; ts.lineH = 17;
    const float th = gx.TextH(text, w - 32, ts);
    gx.Text(text, Rc{x + 16, y + 8, w - 32, th + 2}, ts);
    AddText(Rc{x + 16, y + 8, w - 32, th + 2}, text);
    return 8 + th;
}
float GroupList(Gfx& gx, float x, float y, float w, const std::function<float(float top)>& rows) {
    const float h = Measure(gx, [&] { return rows(y); });
    GroupBox(gx, Rc{x, y, w, h});
    rows(y);
    return h;
}
float HairlineW(const Gfx& gx) {
    const float s = std::max(1.f, gx.scale);
    return std::max(1.f, std::round(s)) / s;  // one device pixel or more: sharp at 150 % (2 px there, as every list's)
}
void Hairline(Gfx& gx, float x, float y, float right) {
    if (right <= x) return;
    const float s = std::max(1.f, gx.scale);
    gx.Fill(Rc{x, std::round(y * s) / s, right - x, HairlineW(gx)}, pal.sep);
}
void RowFill(Gfx& gx, const Rc& rowIn, bool first, bool last, bool pressed) {
    const Rc row = SnapPx(gx, rowIn);  // flush with the cell's snapped edges
    if (pal.hc) { gx.StrokeRound(row.in(1.5f), first || last ? kGroupR - 1.5f : 2.f, pal.tint, 2); return; }  // a Contrast theme: the highlight's edge
    const Col c = pressed ? Mix(pal.cell, pal.fillHover, 0.6f) : Mix(pal.cell, pal.fill, 0.45f);
    gx.FillRound4(row, first ? kGroupR : 0.f, first ? kGroupR : 0.f, last ? kGroupR : 0.f, last ? kGroupR : 0.f, c);
}
static bool SameRgb(Col a, Col b) { return a.r == b.r && a.g == b.g && a.b == b.b; }
void StatusDot(Gfx& gx, float cx, float cy, Col c) {
    if (onArtDepth > 0 && pal.hc) {  // a Contrast theme on art: the system's colour (as off the art), ringed in the art's text colour
        gx.Circle(cx, cy, 4.5f, pal.onArt);
        gx.Circle(cx, cy, 3.5f, c);
        return;
    }
    if (onArtDepth > 0)  // on art: the art's dot of the same meaning
        c = SameRgb(c, pal.green) ? pal.greenArt : SameRgb(c, pal.amber) ? pal.amberArt : SameRgb(c, pal.greyDot) ? pal.greyArt : SameRgb(c, pal.red) ? pal.redArt : c;
    gx.Circle(cx, cy, 3.5f, c);
}
void UpdateMark(Gfx& gx, float cx, float cy, Col words) { gx.Icon(ic::arrowUp, cx, cy, 11, words); }
wstring NoWidow(const wstring& t) {
    const size_t last = t.find_last_of(L' ');
    if (last == wstring::npos || last == 0 || t.size() - last > 12) return t;
    wstring s = t;
    s[last] = L'\u00a0';
    const size_t prev = s.find_last_of(L' ', last - 1);
    if (prev != wstring::npos && last - prev <= 10) s[prev] = L'\u00a0';  // the last line keeps two words at least
    return s;
}
void Chevron(Gfx& gx, float cx, float cy) { gx.Icon(ic::chevron, cx, cy, 11, pal.text2); }
void Led(Gfx& gx, float cx, float cy, Col c, bool, float) { StatusDot(gx, cx, cy, c); }

// A section label: option E's group heading. The first run's numbered steps are not drawn (E has none).
float SectionLabel(Gfx& gx, float x, float y, float w, const wstring& text, int, float trailW) { return GroupHeading(gx, x - 4, y, w + 4, text, L"", trailW); }

// ---- motion -------------------------------------------------------------------------------------------------------------------
// Entrances are short (140 to 240 ms) and ease out (fast start, soft landing). Windows' "Animation effects" switch
// (Accessibility > Visual effects; Apple's Reduce Motion) turns every one of them off: things appear where they end.
// Read once and kept (it is asked many times a frame); WM_SETTINGCHANGE reads it again (MotionRefresh).
static int motionOn = -1;
bool MotionOn() {
    if (motionOn < 0) {
        BOOL motion = TRUE;
        SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, &motion, 0);
        motionOn = motion != FALSE;
    }
    return motionOn != 0;
}
void MotionRefresh() { motionOn = contrastOn = transparencyOn = -1; }
// QueryPerformanceCounter in milliseconds, lined up with GetTickCount64 once (stamps from either can be compared; the tick
// count lags by up to one 15.6 ms tick, so an entrance stamped with it starts at most one tick early)
ULONGLONG NowMs() {
    static const auto base = [] {
        LARGE_INTEGER f{}, c{};
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&c);
        return std::make_tuple(f.QuadPart, c.QuadPart, GetTickCount64());
    }();
    LARGE_INTEGER c{};
    QueryPerformanceCounter(&c);
    const long long freq = std::get<0>(base) > 0 ? std::get<0>(base) : 1;
    return std::get<2>(base) + (ULONGLONG)((c.QuadPart - std::get<1>(base)) * 1000 / freq);
}
float Entrance(ULONGLONG at, float ms) {
    float t = 1;
    if (v.motionAt >= 0) t = std::min(1.f, v.motionAt);
    else if (v.live && at && MotionOn()) t = std::min(1.f, std::max(0.f, (float)((long long)NowMs() - (long long)at)) / ms);
    return 1 - (1 - t) * (1 - t) * (1 - t);  // ease out (cubic)
}
bool PageEntering() { return Entrance(v.pageChangedAt, kPageMs) < 1; }
bool PageUnderSheet() {
    return v.live && ((v.dialog.open && Entrance(v.dialog.at, kDialogMs) >= 1) || (v.first.open && Entrance(v.first.at, kFirstMs) >= 1));
}
// Switches and segmented controls slide to a new state (the mockup's .15 s on a switch's track colour and knob, as Windows'
// and Apple's switches and segmented controls move): the state each one last drew (by id) and when it changed. Only a change
// on the same visit of a page slides (a page that opens shows its states as they are); none offscreen (renders) or with
// Animation effects off (Entrance).
static constexpr float kSlideMs = 150;
struct Slide { int state = 0, from = 0; ULONGLONG at = 0, visit = 0; int page = -1; };
static std::map<int, Slide> slides;
static ULONGLONG slideLast = 0;  // the latest slide's start (MotionPending keeps the frames coming while it moves)
// The overlay scroll thumb grows (5 to 7 DIPs, a little stronger) as the pointer comes onto its track and shrinks as it leaves,
// eased over kThumbGrowMs like the rest, not snapping: when the pointer last came or went (DrawScrollbar)
constexpr float kThumbGrowMs = 120;
static ULONGLONG thumbAt = 0;
// How far control `id`'s slide to `state` has gone (eased; 1 at rest); `from`: the state it slides from (`state` at rest)
static float SlideT(int id, int state, int& from) {
    from = state;
    if (!v.live) return 1;
    Slide& s = slides[id];
    if (s.visit != v.pageChangedAt || s.page != (int)v.page) { s = Slide{state, state, 0, v.pageChangedAt, (int)v.page}; return 1; }
    if (s.state != state) { s.from = s.state; s.state = state; s.at = slideLast = NowMs(); }
    if (!s.at) return 1;
    const float t = Entrance(s.at, kSlideMs);
    if (t >= 1) { s.at = 0; return 1; }
    from = s.from;
    return t;
}
bool MotionPending() {
    if (!v.live || v.motionAt >= 0 || !MotionOn()) return false;
    const ULONGLONG now = NowMs();
    auto moving = [&](ULONGLONG at, float ms) { return at && (long long)now - (long long)at < (long long)ms + 20; };
    if (moving(v.pageChangedAt, kPageMs)) return true;
    if (v.menu.open && moving(v.menu.at, kMenuMs)) return true;
    if (v.dialog.open && moving(v.dialog.at, kDialogMs)) return true;
    if (v.first.open && (moving(v.first.at, kFirstMs) || moving(v.first.resizeAt, kFirstMs))) return true;  // its entrance, a new height
    if (!v.toast.text.empty() && (moving(v.toast.at, kToastMs) || (now + (ULONGLONG)kToastOutMs >= v.toast.until && now < v.toast.until + 20))) return true;
    if (moving(slideLast, kSlideMs)) return true;  // a switch's knob, a segmented control's selection sliding
    if (moving(thumbAt, kThumbGrowMs)) return true;  // the scroll thumb growing or shrinking
    for (const auto& w : v.wipeAt) if (moving(w.second, kWipeMs)) return true;  // the upscale wipe's fade
    for (const auto& w : v.wipeOut) if (moving(w.second, kWipeMs)) return true;  // and its fade out
    if (HeroFading()) return true;  // a banner's hero fading in (ui_art.cpp)
    if (GameMotion()) return true;  // the unsaved capsule's entrance or exit (ui_game.cpp)
    return false;
}

// ---- buttons ----------------------------------------------------------------------------------------------------------------
// A Contrast theme's edge round a control on game art (the glass capsule, the art's red pill): in the art's text colour (the
// system's window text can be dark on the dark scrim), 2 DIPs under the pointer, faded when off; nothing in the other themes
void ArtHcEdge(Gfx& gx, const Rc& r, float rad, bool hot, bool enabled) {
    if (!pal.hc) return;
    const float w = hot ? 2.f : 1.f;
    gx.StrokeRound(r.in(w / 2), std::max(0.f, rad - w / 2), Alpha(pal.onArt, enabled ? 1.f : 0.55f), w);
}
static float BtnFont(float h) { return h >= 34 ? 14.f : 13.f; }
static float BtnPad(BtnKind k, float h) { return k == B_LINK ? 6.f : k == B_GHOST ? kGhostPad : h >= 34 ? 18.f : 14.f; }
float ButtonW(Gfx& gx, const wstring& text, wchar_t icon, float h, BtnKind kind) {
    if (text.empty()) return std::ceil(std::max(h, 28.f));  // an icon alone: a round (or square) button
    float w = BtnPad(kind, h) * 2 + gx.TextW(text, Ts(BtnFont(h), 600, pal.text));
    if (icon) w += 13 + 8;
    return std::ceil(w);
}

float Button(Gfx& gx, int id, float x, float y, const wstring& text, BtnKind kind, wchar_t icon, bool enabled, float h, float minW, bool alignRight,
             const wstring& name, float rad) {
    const float w = std::max(minW, ButtonW(gx, text, icon, h, kind));
    if (alignRight) x -= w;
    const Rc r{x, y, w, h};
    if (rad < 0) rad = kind == B_PLAIN || kind == B_GHOST ? 8.f : kind == B_LINK ? 6.f : h / 2;
    const float fs = BtnFont(h);
    const bool hot = enabled && Hot(id), down = enabled && Down(id);
    Col fg = pal.text, ig = pal.text;
    if (pal.hc && kind != B_GLASS && kind != B_ART_TINTED) {
        // a Contrast theme: the filled ones (Play, Apply, Continue) and a pressed one on the highlight in its text colour; every
        // other shape an edge (a link's and a ghost's only under the pointer), the highlight's under the pointer; off: gray
        const bool filled = enabled && (kind == B_PRIMARY || kind == B_GO || down);
        if (filled) gx.FillRound(r, rad, pal.tint);
        if (filled && hot && !down) gx.StrokeRound(r.in(2.5f), std::max(0.f, rad - 2.5f), pal.onAccent, 1.5f);  // under the pointer: an inner ring
        if ((kind != B_LINK && kind != B_GHOST) || hot) HcEdge(gx, r, rad, hot && !filled, enabled);
        fg = ig = !enabled ? pal.text3 : filled ? pal.onAccent : kind == B_LINK ? pal.tintText : pal.text;
    } else switch (kind) {
        case B_PRIMARY: {  // hover and pressed darker, as B_GO (white on the accent 5.7:1, more when darker)
            Col f = pal.accent;
            if (hot) f = Mix(f, pal.black, 0.08f);
            if (down) f = Mix(pal.accent, pal.black, 0.16f);
            // disabled: the one rule of every filled button (ui_palette.h DisabledFilled: faded, its label 3.5:1)
            const FilledOff off = DisabledFilled(pal.accent, pal.cell, pal.onAccent, pal.dark);
            if (!enabled) f = off.fill;
            gx.FillRound(r, rad, f);
            fg = ig = enabled ? pal.onAccent : off.label;
            break;
        }
        case B_GLASS:  // white at 18 % over the banner's dark scrim (the mockup's glass), lighter under the pointer
            gx.FillRound(r, rad, Alpha(pal.white, down ? 0.32f : hot ? 0.26f : 0.18f));
            ArtHcEdge(gx, r, rad, hot, enabled);
            fg = ig = Alpha(pal.onArt, enabled ? 1.f : 0.55f);
            break;
        case B_SECONDARY: case B_PLAIN:
            gx.FillRound(r, rad, down ? Mix(pal.fillHover, pal.text, 0.06f) : hot ? pal.fillHover : pal.fill);
            fg = ig = enabled ? pal.text : pal.text3;
            break;
        case B_GO: {  // white on the accent's fill 4.5:1 or better; hover and pressed darker, disabled toward the cell
            Col f = pal.tintFill;
            if (hot) f = Mix(f, pal.black, 0.08f);
            if (down) f = Mix(pal.tintFill, pal.black, 0.16f);
            // disabled: the same rule as the red one (B_PRIMARY), so Play, Continue and Update N games read off alike
            const FilledOff off = DisabledFilled(pal.tintFill, pal.cell, pal.onAccent, pal.dark);
            if (!enabled) f = off.fill;
            gx.FillRound(r, rad, f);
            fg = ig = enabled ? pal.onAccent : off.label;
            break;
        }
        case B_TINTED:  // accentText on the tint: 5.4:1 (light), 6.5:1 (dark)
            gx.FillRound(r, rad, down ? Mix(pal.accentTintHover, pal.accent, 0.08f) : hot ? pal.accentTintHover : pal.accentTint);
            fg = ig = enabled ? pal.accentText : pal.text3;
            break;
        case B_ART_TINTED:  // opaque on the art (its words never depend on it): 6.0:1; disabled 3.2:1
            gx.FillRound(r, rad, down ? Mix(pal.artTintHover, pal.artTintText, 0.08f) : hot ? pal.artTintHover : pal.artTint);
            ArtHcEdge(gx, r, rad, hot, enabled);
            fg = ig = enabled ? pal.artTintText : Mix(pal.artTint, pal.artTintText, 0.62f);
            break;
        case B_GHOST:
            if (hot) gx.FillRound(r, rad, down ? pal.fillHover : pal.fill);
            fg = enabled ? pal.text : pal.text3;
            ig = enabled ? pal.text2 : pal.text3;
            break;
        case B_LINK:  // no underline (a web idiom): the fill under the mouse, as the ghost buttons have it; the tint's words
            if (hot) gx.FillRound(r, rad, down ? pal.fillHover : pal.fill);
            fg = ig = enabled ? pal.tintText : pal.text3;
            break;
    }
    TS ts = Ts(fs, 600, fg); ts.valign = 1;
    if (text.empty()) {
        if (icon) gx.Icon(icon, r.cx(), r.cy(), 13, ig);
    } else {
        const float tw = gx.TextW(text, ts);
        float cx = r.x + (r.w - tw - (icon ? 21.f : 0.f)) / 2;
        if (kind == B_GHOST || kind == B_LINK) cx = std::min(cx, r.x + BtnPad(kind, h));  // a wider ghost keeps its label on the text edge
        if (kind == B_LINK && alignRight) cx = r.r() - BtnPad(kind, h) - tw - (icon ? 21.f : 0.f);  // its words end on the column's edge
        if (icon) { gx.Icon(icon, cx + 6.5f, r.cy(), 13, ig); cx += 21; }
        gx.Text(text, Rc{cx, r.y, tw + 2, r.h}, ts);
    }
    FocusRing(gx, r, rad, id);
    AddHit(id, r, R_BUTTON, !name.empty() ? name : text.empty() ? L"More" : text, enabled);
    return w;
}

float Pill(Gfx& gx, int id, float right, float cy, const wstring& text, bool solid, bool enabled, const wstring& name) {
    return Button(gx, id, right, std::round(cy - kPillH / 2), text, solid ? B_GO : B_TINTED, 0, enabled, kPillH, kPillMinW, true, name);
}

// A switch: a 46 x 28 capsule track (the accent when on: 3:1 or better on a cell), a 22 DIP white knob with a small shadow.
// A new state slides: the knob travels and the track's colour follows it (SlideT).
void Switch(Gfx& gx, int id, const Rc& r, bool on, bool enabled, const wstring& name) {
    const Rc t{std::round(r.r() - kSwitchW), std::round(r.cy() - kSwitchH / 2), kSwitchW, kSwitchH};
    int from = 0;
    const float st = SlideT(id, on ? 1 : 0, from), p = on ? st : 1 - st;  // 0 off .. 1 on, where the knob is now
    const float kx = t.x + 14 + (t.w - 28) * p;
    if (pal.hc) {  // a Contrast theme: on, the highlight with a knob in its text colour; off, an edge and a knob in the text colour
        if (on && enabled) gx.FillRound(t, kSwitchH / 2, pal.tint);
        HcEdge(gx, t, kSwitchH / 2, Hot(id) && enabled && !on, enabled);
        if (on && enabled && Hot(id)) gx.StrokeRound(t.in(2.5f), kSwitchH / 2 - 2.5f, pal.onAccent, 1.5f);
        gx.Circle(kx, t.cy(), 7 + 2 * p, !enabled ? pal.text3 : on ? pal.onAccent : pal.text);
        FocusRing(gx, t, kSwitchH / 2, id);
        AddHit(id, t, R_TOGGLE, name, enabled, on ? 1 : 0);
        return;
    }
    Col track = Mix(pal.trackOff, pal.tint, p);
    if (!enabled) track = Mix(track, pal.cell, 0.5f);
    else if (Down(id)) track = Mix(track, pal.text, 0.16f);  // pressed: darker again, as every control has it
    else if (Hot(id)) track = Mix(track, pal.text, 0.07f);
    gx.FillRound(t, kSwitchH / 2, track);
    // off: a 1 DIP edge, so the track's shape stands out from the cell (3:1); the knob's place is the second cue
    if (p < 1) gx.StrokeRound(t.in(0.5f), kSwitchH / 2 - 0.5f, Alpha(enabled ? pal.trackEdge : Mix(pal.trackEdge, pal.cell, 0.5f), 1 - p), 1);
    const float ky = t.cy();
    gx.Circle(kx, ky + 1, 11.5f, Alpha(pal.black, enabled ? 0.18f : 0.08f));  // 0 1px 3px rgba(0,0,0,.3)
    gx.Circle(kx, ky, 11, enabled ? pal.knob : Mix(pal.knob, pal.cell, 0.3f));
    FocusRing(gx, t, kSwitchH / 2, id);
    AddHit(id, t, R_TOGGLE, name, enabled, on ? 1 : 0);
}

// Segments: 28 high, 2 apart, in a well of the fill 2 DIPs round them (32 high); the selected one raised on segOn.
static constexpr float kSegPad = 10, kSegMin = 58, kSegFont = 12.5f;
float Segmented(Gfx& gx, int idBase, float x, float y, const std::vector<wstring>& opts, int sel, bool enabled, const wstring& name, float h,
                const std::vector<wstring>& counts, bool alignRight, float minSeg) {
    const float sh = std::max(h, 28.f), wellH = sh + 4;
    std::vector<float> ws;
    for (size_t i = 0; i < opts.size(); ++i) {
        float w = kSegPad * 2 + gx.TextW(opts[i], Ts(kSegFont, 600, pal.text));
        if (i < counts.size() && !counts[i].empty()) w += 5 + gx.TextW(counts[i], Ts(kSegFont, 400, pal.text2));
        ws.push_back(std::ceil(std::max({kSegMin, minSeg, w})));
    }
    // equal segments (Apple HIG, Segmented controls: "all segments are usually equal in width"); with counts they stay
    // sized to their content
    if (counts.empty() && !ws.empty()) { const float m = *std::max_element(ws.begin(), ws.end()); for (auto& w : ws) w = m; }
    float total = 4;
    for (size_t i = 0; i < ws.size(); ++i) total += ws[i] + (i ? 2.f : 0.f);
    if (alignRight) x -= total;
    const Rc box{x, y + std::round((h + 8 - wellH) / 2), total, wellH};
    gx.FillRound(box, 8, pal.fill);
    HcEdge(gx, box, 8, false, enabled);
    // the chosen one's raised thumb, under every label: a new choice slides it from the old one's place (SlideT)
    const int n = (int)opts.size();
    if (sel >= 0 && sel < n) {
        std::vector<float> xs;
        float px = x + 2;
        for (int i = 0; i < n; ++i) { xs.push_back(px); px += ws[i] + 2; }
        int from = sel;
        const float st = SlideT(idBase, sel, from);
        Rc b{xs[sel], box.y + 2, ws[sel], sh};
        if (st < 1 && from >= 0 && from < n && from != sel) { b.x = xs[from] + (xs[sel] - xs[from]) * st; b.w = ws[from] + (ws[sel] - ws[from]) * st; }
        const bool down = Down(idBase + sel) && enabled;
        if (pal.hc) {  // a Contrast theme: the chosen one on the highlight (gray edge when off)
            if (enabled) gx.FillRound(b, 6, pal.tint);
            else gx.StrokeRound(b.in(1), 5, pal.text3, 2);
        } else {
            gx.FillRound(b.off(0, 1), 6, Alpha(pal.black, enabled ? 0.10f : 0.04f));  // 0 1px 2px rgba(0,0,0,.14)
            gx.FillRound(b, 6, !enabled ? Mix(pal.segOn, pal.fill, 0.4f) : down ? Mix(pal.segOn, pal.fillHover, 0.5f) : pal.segOn);
        }
    }
    float bx = x + 2;
    for (size_t i = 0; i < opts.size(); ++i) {
        const int id = idBase + (int)i;
        const Rc b{bx, box.y + 2, ws[i], sh};
        const bool on = (int)i == sel, hot = Hot(id) && enabled, down = Down(id) && enabled;
        // the others under the pointer or pressed (the chosen one's thumb is drawn above); a Contrast theme: an edge
        if (!on && pal.hc) { if (hot || down) HcEdge(gx, b, 6, true); }
        else if (!on && down) gx.FillRound(b, 6, Mix(pal.fillHover, pal.text, 0.06f));
        else if (!on && hot) gx.FillRound(b, 6, pal.fillHover);
        // disabled: labels in text3 (3:1 or better on the well, POLISH-AUDIT.md #33); the chosen one on its dimmed thumb a
        // step lighter in dark (text3 there is 2.6:1; this 3.5:1), so the value chosen stays the clearest one to read
        const Col c = pal.hc ? (!enabled ? pal.text3 : on ? pal.onAccent : pal.text)
                             : !enabled ? (on && pal.dark ? Mix(pal.text3, pal.text2, 0.6f) : pal.text3) : on || hot ? pal.text : pal.text2;
        TS ts = Ts(kSegFont, 600, c); ts.valign = 1;
        const bool cnt = i < counts.size() && !counts[i].empty();
        const float tw = gx.TextW(opts[i], ts), cw = cnt ? 5 + gx.TextW(counts[i], Ts(kSegFont, 400, c)) : 0;
        const float tx = cnt ? b.x + kSegPad : std::round(b.x + (b.w - tw) / 2);
        gx.Text(opts[i], Rc{tx, b.y, tw + 2, b.h}, ts);
        if (cnt) { TS cs = Ts(kSegFont, 400, c); cs.valign = 1; gx.Text(counts[i], Rc{tx + tw + 5, b.y, cw + 2, b.h}, cs); }
        FocusRing(gx, b, 6, id);
        AddHit(id, b, R_RADIO, name + L": " + opts[i] + (cnt ? L" (" + counts[i] + L")" : L""), enabled, on ? 1 : 0);
        bx += ws[i] + 2;
    }
    return total;
}

float SegmentW(Gfx& gx, const std::vector<wstring>& labels) {
    float m = kSegMin;
    for (auto& o : labels) m = std::max(m, std::ceil(kSegPad * 2 + gx.TextW(o, Ts(kSegFont, 600, pal.text))));
    return m;
}
float SegmentedW(Gfx& gx, const std::vector<wstring>& opts, float minSeg) {
    const float m = std::max(minSeg, SegmentW(gx, opts));
    return 4 + m * opts.size() + 2 * (opts.size() - 1);
}

// Filter chips: 30 high capsules 4 apart; the selected one on the fill in the text colour, the others text2 (5.6:1 on the
// page). A count follows its label in the regular weight.
float Tabs(Gfx& gx, int idBase, float x, float y, const std::vector<wstring>& opts, int sel, const wstring& name, const std::vector<wstring>& counts, bool measureOnly) {
    float tx = x;
    for (size_t i = 0; i < opts.size(); ++i) {
        const int id = idBase + (int)i;
        const bool on = (int)i == sel, hot = Hot(id);
        const Col c = pal.hc && on ? pal.onAccent : on || hot ? pal.text : pal.text2;
        TS ts = Ts(13, 600, c); ts.valign = 1;
        TS cs = Ts(13, 400, c); cs.valign = 1;
        const wstring cnt = i < counts.size() ? counts[i] : L"";
        const float tw = gx.TextW(opts[i], ts), cw = cnt.empty() ? 0 : gx.TextW(cnt, cs) + 6;
        const Rc r{tx, y, std::ceil(12 + tw + cw + 12), 30};
        if (!measureOnly) {
            if (pal.hc) {  // a Contrast theme: the chosen chip on the highlight, the pointer's an edge
                if (on) gx.FillRound(r, 15, pal.tint);
                else if (hot || Down(id)) HcEdge(gx, r, 15, true);
            } else if (Down(id)) gx.FillRound(r, 15, pal.fillHover);
            else if (on) gx.FillRound(r, 15, pal.fill);
            gx.Text(opts[i], Rc{r.x + 12, r.y, tw + 2, r.h}, ts);
            if (!cnt.empty()) gx.Text(cnt, Rc{r.x + 12 + tw + 6, r.y, cw, r.h}, cs);
            FocusRing(gx, r, 15, id);
            AddHit(id, r, R_RADIO, name + L": " + opts[i] + (cnt.empty() ? L"" : L" (" + cnt + L")"), true, on ? 1 : 0);
        }
        tx += r.w + 4;
    }
    return tx - 4 - x;
}

void Select(Gfx& gx, int id, const Rc& r, const wstring& text, bool enabled, const wstring& name) {
    const bool hot = Hot(id) && enabled, open = v.menu.open && v.menu.owner == id;
    gx.FillRound(r, 8, enabled && (Down(id) || open) ? Mix(pal.fillHover, pal.text, 0.06f) : hot ? pal.fillHover : pal.fill);  // pressed while its menu is open
    HcEdge(gx, r, 8, hot || open, enabled);
    TS ts = Ts(13, 600, enabled ? pal.text : pal.text3); ts.valign = 1;
    gx.Text(text, Rc{r.x + 12, r.y, r.w - 12 - 30, r.h}, ts);
    gx.Icon(ic::down, r.r() - 16, r.cy() + 1, 10, enabled ? pal.text2 : pal.text3);
    FocusRing(gx, r, 8, id);
    AddHit(id, r, R_COMBO, name, enabled, -1, text);
}

float Chip(Gfx& gx, float x, float cy, const wstring& text, Col fg, Col border, Col bg) {
    TS ts = Ts(11.5f, 600, fg); ts.valign = 1;
    const float tw = gx.TextW(text, ts);
    const Rc r{x, std::round(cy - 10), std::ceil(tw + 14), 20};
    if (bg.a > 0) gx.FillRound(r, 5, bg);
    if (border.a > 0 || pal.hc) gx.StrokeRound(r, 5, pal.hc ? pal.edge : border);
    gx.Text(text, Rc{r.x + 7, r.y, tw + 2, r.h}, ts);
    return r.w;
}

// A tag: small words on the fill (the mockup's .tag: "+0.11 ms", "WrongGPU's setup"), in their own colour.
float Tag(Gfx& gx, float x, float cy, const wstring& text, Col fg, Col) { return Chip(gx, x, cy, text, fg, Col{0, 0, 0, 0}, pal.fill); }

// A plain-language note in a quiet box tinted with its tone (0 needs a look, 1 good, 2 problem, 3 information); `link`
// (e.g. "Dismiss") is a clickable word after the text (id). A problem (an operation that didn't finish) is an error in words:
// red (IDENTITY.md), its glyph accentText on the red tint (5.6:1 light, 6.2:1 dark); a look is amber, a warning.
float Callout(Gfx& gx, int id, float x, float y, float w, const wstring& text, int tone, const wstring& link, wchar_t icon) {
    const Col tc = tone == 1 ? pal.green : tone == 3 ? pal.text2 : tone == 2 ? pal.accent : pal.amber;
    TS ts = Ts(13, 400, pal.text); ts.wrap = true; ts.lineH = 18;
    const wstring full = link.empty() ? text : text + L"  " + link;
    const float avail = w - 40 - 16;
    const float th = gx.TextH(full, avail, ts);
    const Rc r{x, y, w, std::max(40.f, th + 22)};
    // on the page: information is a plain cell (a tint of grey would be the page's own colour), the others tinted cells
    // a Contrast theme: no tint (the glyph and the words say the tone), an edge; the lists' radius
    if (tone == 3 || pal.hc) GroupBox(gx, r, kGroupR);
    else gx.FillRound(r, kGroupR, Mix(pal.cell, tc, pal.dark ? 0.12f : 0.08f));
    gx.Icon(icon ? icon : tone == 1 ? ic::check : tone == 3 ? ic::info : ic::warn, r.x + 21, r.y + 20, 13,
            tone == 1 ? pal.greenText : tone == 3 ? pal.text2 : tone == 2 ? pal.accentText : pal.amberText);
    gx.Text(text, Rc{r.x + 40, r.y + 11, avail, th + 4}, ts);
    AddText(Rc{r.x + 40, r.y + 11, avail, th + 4}, text);  // before its link: the link takes the click
    if (!link.empty()) {
        // the link follows the text's last line, or goes on a line of its own; in the text colour (the accent is for the
        // next thing to press, not a red word on a green or amber note), the fill under the mouse
        TS ls = ts; ls.color = pal.text; ls.wrap = false; ls.weight = 600;
        const float lw = gx.TextW(link, ls);
        float ex = 0, ey = 0;
        gx.TextEnd(text, avail, ts, ex, ey);
        float lx = r.x + 40 + ex + 8, ly = r.y + 11 + ey;
        if (ex + 8 + lw > avail) { lx = r.x + 40; ly += 18; }
        const Rc lr{lx - 6, ly - 5, lw + 12, 28};  // a 28 DIP target around the 18 DIP line
        if (Hot(id)) { if (pal.hc) HcEdge(gx, lr, 6, true); else gx.FillRound(lr, 6, Alpha(pal.text, Down(id) ? 0.12f : 0.07f)); }
        gx.Text(link, Rc{lx, ly, lw + 4, 18}, ls);
        FocusRing(gx, lr, 6, id);
        AddHit(id, lr, R_BUTTON, link, true);  // a button: it acts in place ("Dismiss"), a link goes somewhere
        Describe(id, text);  // "Dismiss" says what it dismisses
        if (ly + 18 > r.b() - 8) return r.h + 18;
    }
    return r.h;
}

// A text field on the fill (the search, the release address, OptiScaler numbers); the native EDIT child sits on it
// (offscreen renders draw the text here). Focused: the focus ring.
void FieldBox(Gfx& gx, EditBox& e, int id, const Rc& r, const wstring& text, const wstring& cue, wchar_t icon, const wstring& name, bool mono, int clearId,
              bool enabled) {
    // the ring: the native field has the keys, or the keyboard is on its way to it (Tab to a field still scrolling into view)
    const bool focused = enabled && ((e.h && GetFocus() == e.h) || Focused(id));
    const float rad = r.h >= 32 ? 9.f : 8.f;
    gx.FillRound(r, rad, pal.fill);
    HcEdge(gx, r, rad, false, enabled);
    // a value Apply refused: a red edge (accentText: 4.5:1 or better on the fill) until its text changes (FieldInvalidWhy)
    if (enabled && !FieldInvalidWhy(id).empty()) gx.StrokeRound(r.in(0.75f), rad - 0.75f, pal.accentText, 1.5f);
    if (focused) gx.StrokeRound(r.in(-4), rad + 4, pal.focus, 2);
    float tx = r.x + 12;
    if (icon) { gx.Icon(icon, r.x + 12 + 6.5f, r.cy(), 13, pal.text2); tx = r.x + 12 + 13 + 8; }
    const bool clear = clearId && !text.empty();
    const float right = clear ? 32.f : 10.f;
    e.r = Rc{tx, r.y + 1, r.r() - tx - right, r.h - 2}.off(0, -cv.scroll);
    e.view = cv.view;
    e.want = true;
    e.cue = cue;
    e.name = name;
    e.enabled = enabled;
    // no live EDIT (offscreen renders), or it is hidden (the page's entrance, which it cannot fade; the field partly under the
    // toolbar, the pinned header or the unsaved band; a menu over it): the text drawn here, clipped with the page
    if (!e.h || gx.offscreen || !EditShown(e)) {
        TS ts = Ts(13, 400, !enabled ? pal.text3 : text.empty() ? pal.text2 : pal.text); ts.valign = 1;  // disabled: text3 (3:1 or better)
        if (mono) { ts.face = F_MONO; ts.size = 12; }
        gx.Text(text.empty() ? cue : text, Rc{tx, r.y, r.r() - tx - right, r.h}, ts);
    }
    AddHit(id, Rc{r.x, r.y, r.w - (clear ? 32.f : 0.f), r.h}, R_EDIT, name, enabled, -1, text);
    if (clear) {  // 28 DIPs, a stop for Tab after the field (Enter or Space clears it and the field takes the keys again)
        const Rc cr{r.r() - 30, std::round(r.cy() - 14), 28, 28};
        if (Hot(clearId)) gx.FillRound(cr, 14, Down(clearId) ? Mix(pal.fillHover, pal.text, 0.06f) : pal.fillHover);
        if (Hot(clearId)) HcEdge(gx, cr, 14, true);
        gx.Icon(ic::close, cr.cx(), cr.cy(), 10, Hot(clearId) || Focused(clearId) ? pal.text : pal.text2);
        FocusRing(gx, cr, 14, clearId);
        AddHit(clearId, cr, R_BUTTON, L"Clear search", true);
    }
}

bool EditShown(const EditBox& e) {
    const Rc view = e.view.w > 0 ? e.view : Rc{0, kBarH, v.W, v.H - kBarH};
    const Rc r = e.r;
    if (!e.want || !e.enabled || v.dialog.open || v.first.open || PageEntering()) return false;
    if (r.y < std::max(kBarH, view.y) - 1 || r.b() > view.b() + 1 || r.x < view.x - 1 || r.r() > view.r() + 1) return false;
    if (v.menu.open) {  // under an open menu: hidden
        const Rc mr = MenuRect();
        if (Intersect(mr, r).w > 0 && Intersect(mr, r).h > 0) return false;
    }
    return true;
}

// ---- covers -----------------------------------------------------------------------------------------------------------------
wstring Monogram(const wstring& name) {
    std::vector<wstring> words;
    wstring w;
    for (wchar_t c : name + L" ") {
        if (iswalnum(c)) w += c;
        else if (c == L'\'' || c == L'’') continue;  // "Clancy's" stays one word
        else if (!w.empty()) { words.push_back(w); w.clear(); }
    }
    if (words.size() > 1 && (IEq(words[0], L"the") || IEq(words[0], L"a") || IEq(words[0], L"an"))) words.erase(words.begin());
    wstring m;
    for (size_t i = 0; i < words.size() && m.size() < 2; ++i) m += (wchar_t)towupper(words[i][0]);
    return m;
}

// A game without Steam art: a quiet two-stop gradient in the name's hue (the mockup's 160deg #20343a to #0b1214) and its
// initials in white at 92 %.
void CoverPlate(Gfx& gx, const Rc& r, const wstring& name, float rad) {
    Col accent, dark; NameColours(name, accent, dark);
    gx.Grad(r, rad, Mix(dark, accent, 0.12f), Mix(dark, pal.black, 0.6f), 0.33f, 0, 0.67f, 1);
    TS ms = Ts(std::round(r.w * 0.34f), 700, Alpha(pal.white, 0.92f), F_DISPLAY); ms.align = 1; ms.valign = 1;
    ms.spacing = r.w * 0.014f;
    gx.Text(Monogram(name), r, ms);
}

void Cover(Gfx& gx, const Rc& r, const wstring& appid, const wstring& name, float rad, bool shadow) {
    const bool big = r.w >= 60;
    // rows: 0 1px 3px rgba(0,0,0,.18); the game page: 0 6px 18px rgba(0,0,0,.18) (dark .5)
    if (shadow) gx.Shadow(r, rad, big ? 18.f : 3.f, big ? (pal.dark ? 0.7f : 0.3f) : 0.28f, big ? 6.f : 1.f, true);  // the art or plate covers it
    const Art a = appid.empty() ? Art{} : ArtFor(appid);
    if (a.cover) gx.ImageCover(a.cover, r, rad, 0.5f);
    else CoverPlate(gx, r, name, rad);
}

// ---- the toolbar ------------------------------------------------------------------------------------------------------------
// The mark (the brand icon, assets/wronggpu.ico): a black rounded square with a faint edge, a white "W" out of register (drawn
// first 1 DIP to the left in red and 1 DIP to the right in the working green: the unlock, from wrong to right), and under it
// a bar, fully red (wrong). On the page of a game where WrongGPU is installed and healthy it is "RightGPU": an "R" over a
// fully green bar. `fill`: how much of the bar is green, from the left (the flip as an install finishes); `cross`: the
// letter's cross-fade from W (0) to R (1). Drawn at 24 DIPs in the toolbar and 36 on the first-run sheet; the same
// proportions at any size and in both themes (the mark is always dark).
void AppMark(Gfx& gx, const Rc& m, float fill, float cross) {
    const float k = m.h / 24;
    gx.FillRound(m, 6 * k, Hex(0x000000));
    gx.StrokeRound(m, 6 * k, Alpha(pal.white, 0.14f));
    auto letter = [&](const wchar_t* t, float a) {
        if (a <= 0.01f) return;
        TS ms = Ts(13 * k, 700, Hex(0xff4a3d, a), F_DISPLAY); ms.align = 1; ms.valign = 1;
        gx.Text(t, m.off(-1 * k, -1.5f * k), ms);
        ms.color = Hex(0x4bd865, a);
        gx.Text(t, m.off(1 * k, -1.5f * k), ms);
        ms.color = Hex(0xf2f2f4, a);
        gx.Text(t, m.off(0, -1.5f * k), ms);
    };
    letter(L"W", 1 - cross);
    letter(L"R", cross);
    const Rc bar{m.x + 7 * k, m.b() - 6 * k, m.w - 14 * k, 2 * k};
    gx.FillRound(bar, 1 * k, Hex(0xe0392b));
    if (fill > 0) {
        gx.PushClip(Rc{bar.x, bar.y - 1, bar.w * std::min(1.f, fill), bar.h + 2});
        gx.FillRound(bar, 1 * k, Hex(0x4bd865));
        gx.PopClip();
    }
}

// RightGPU: a game where WrongGPU is installed and healthy (nothing needs a repair or an update, nothing set up outside the
// launcher, and its last run, if any, did not fall back to AMD FSR)
bool RightGpu(const GameEntry& e) {
    if (!e.hasRecord || e.adoptable || (e.status != GameStatus::Installed && e.status != GameStatus::Running)) return false;
    if (!e.needs.empty() && !e.needsFound) return false;  // the game needs a tool of the player's (REFramework) that is gone
    // (a run of Ray Reconstruction alone, r4 #1: the RR games' front end logs nothing, so no runtime "ready" line there; or RR
    // upscaling where Model M didn't, r6 #1. An RR attempt without a successful frame is no healthy run, as Model M that didn't
    // start isn't: r6 #2)
    return !(e.last.found && (e.last.upscaler == L"amd" || (!e.last.ready && !(e.last.rrOnly && e.last.upscaled) && !e.last.rrUpscaled)));
}
// The flip as an install, a repair or an update finishes on the game's page (owner: no glitch): the progress run ends
// (kFlipDelayMs), the bar fills green from the left (kFlipFillMs, ease-out), then W -> R and the wordmark cross-fade
// (kFlipCrossMs). Animation effects off: at once. Later visits: RightGPU from the start.
constexpr float kFlipDelayMs = 120, kFlipFillMs = 300, kFlipCrossMs = 200, kFlipMs = kFlipDelayMs + kFlipFillMs + kFlipCrossMs;
bool BrandNow(float& fill, float& cross) {
    fill = cross = 0;
    if (v.page != P_GAME || !RightGpu(v.cur)) return false;
    fill = cross = 1;
    if (v.flipDir.empty() || !IEq(v.flipDir, v.cur.dir)) return false;
    const float ms = v.flipT >= 0 ? v.flipT * kFlipMs : v.live && v.flipAt && MotionOn() ? (float)((long long)NowMs() - (long long)v.flipAt) : kFlipMs;
    const float f = std::min(1.f, std::max(0.f, (ms - kFlipDelayMs) / kFlipFillMs));
    fill = 1 - (1 - f) * (1 - f) * (1 - f);
    cross = std::min(1.f, std::max(0.f, (ms - kFlipDelayMs - kFlipFillMs) / kFlipCrossMs));
    return ms < kFlipMs;
}

// ---- Signal's registration offset and the upscale wipe ------------------------------------------------------------------------
// A large page title (Library, Settings, the first run's welcome): plain words. The registration offset stays on the mark,
// the keyboard ring on art and progress runs; on a title it read as a blurred render at body sizes (UI review, 0.6.0).
void PageTitle(Gfx& gx, const wstring& text, const Rc& r, const TS& s) {
    gx.Text(text, r, s);
    AddText(Rc{r.x, r.y, std::min(r.w, gx.TextW(text, s) + 2), r.h}, text, 0, 1);  // the page's heading
}

void ProgressRun(Gfx& gx, const Rc& r) {
    if (r.w <= 0) return;
    gx.GradStops(r, std::min(r.h, r.w) / 2, {{0.f, pal.accent}, {1.f, pal.green}}, 0, 0.5f, 1, 0.5f);
}

// The progress bars' motion (the light along a run, the segment without an amount) while the window is the active one: at
// the display's rate (Animating), as Windows' and macOS's own bars move. A wide bar (the pinned header's spans the reading
// column) travels about 1000 DIPs a second, so any fixed step jumps visibly (33 ms was 33 DIPs at a time, held for 4-5
// refreshes at 144 Hz). The cost stays bounded: one frame per refresh, Present paces it (at most one frame queued), and
// only while the player looks at the active window. Not the active one, the light rests and the segment steps at
// kBarStepIdleMs (timer 9); covered (nothing on screen) it only looks now and then whether it shows again.
constexpr unsigned kBarStepIdleMs = 100, kBarStepHiddenMs = 250;
static void BarStep(Gfx& gx, unsigned ms) {
    if (!v.live || !v.hwnd) return;
    if (gx.occluded) ms = kBarStepHiddenMs;
    else if (v.active) { v.barMoving = true; return; }  // the next frame at the display's rate
    SetTimer(v.hwnd, 9, (UINT)(ms - NowMs() % ms), nullptr);  // WM_TIMER 9 asks for the next frame
}
void ProgressBar(Gfx& gx, const Rc& bar, float fraction, bool active) {
    const float rad = bar.h / 2;
    gx.FillRound(bar, rad, pal.fill);
    if (pal.hc) gx.StrokeRound(bar.in(-0.5f), rad + 0.5f, pal.edge, 1);  // a Contrast theme: the track's edge (its fill is the window's)
    if (!active || gx.dry) return;
    const bool moving = v.live && MotionOn();
    if (fraction > 0) {
        // the amount shown glides to a new one (about 0.15 s to most of the way, by the clock: the same at 60 and 144 Hz);
        // one operation runs at a time, so every bar drawing it shares the glide. A smaller amount (a new operation) is
        // taken at once.
        static float shown = 0;
        static ULONGLONG at = 0;
        const float f = std::min(1.f, fraction);
        const ULONGLONG now = NowMs();
        if (!moving || f < shown || now - at > 1000) shown = f;
        else shown += (f - shown) * (1 - std::exp(-(float)(now - at) / 70.f));
        if (std::fabs(f - shown) < 0.002f) shown = f;
        at = now;
        if (moving && shown != f) v.barMoving = true;  // the glide's frames follow at the display's rate (Animating)
        const Rc run{bar.x, bar.y, std::max(bar.h, bar.w * shown), bar.h};
        ProgressRun(gx, run);
        if (moving && v.active) {
            // a long step (copying files, a conversion) moves the amount rarely: a soft light travels along the run meanwhile,
            // so it never looks stuck (at the display's rate: BarStep)
            const float lw = std::min(80.f, run.w * 0.6f);
            const float t = (float)(now % 1600) / 1600.f;
            const float lx = run.x - lw + (run.w + lw) * t;
            gx.PushRoundClip(run, rad);
            gx.GradStops(Rc{lx, run.y, lw, run.h}, 0, {{0.f, Alpha(pal.white, 0)}, {0.5f, Alpha(pal.white, pal.dark ? 0.22f : 0.35f)}, {1.f, Alpha(pal.white, 0)}}, 0, 0.5f, 1, 0.5f);
            gx.PopRoundClip();
            BarStep(gx, kBarStepIdleMs);
        }
        return;
    }
    if (!MotionOn()) {
        // Animation effects off: no segment standing at the left (it would read as "30 % done"), the whole run dimmed:
        // working, no amount
        gx.PushFade(0.4f);
        ProgressRun(gx, bar);
        gx.PopFade();
        return;
    }
    // the segment travels (a still frame offscreen: caught in the middle of the track, never at its start)
    const float sw = std::round(bar.w * 0.3f);
    const float x = moving ? bar.x - sw + (bar.w + sw) * (float)(NowMs() % 1400) / 1400.f : std::round(bar.x + (bar.w - sw) / 2);
    if (moving) BarStep(gx, kBarStepIdleMs);
    gx.PushRoundClip(bar, rad);
    ProgressRun(gx, Rc{x, bar.y, sw, bar.h});
    gx.PopRoundClip();
}

void BusyIndicator(Gfx& gx, float cx, float cy, float r, Col c) {
    if (gx.dry) return;
    // it turns with Animation effects off too: it is status, not decoration (nothing moves across the screen; Windows'
    // ProgressRing and Reduce Motion on Apple's platforms keep activity indicators turning), and still it would look stuck
    const bool moving = v.live;
    const ULONGLONG now = NowMs();
    const int lead = moving ? (int)((now / 90) % 8) : 0;
    const float w = std::max(1.5f, r * 0.3f);
    for (int i = 0; i < 8; ++i) {
        const float a = 3.14159265f * 2 * i / 8 - 3.14159265f / 2, ca = std::cos(a), sa = std::sin(a);
        const int age = (lead - i + 8) % 8;  // the leading spoke darkest, the ones behind it fading
        const float alpha = moving ? 1.f - age * 0.11f : 0.35f + 0.65f * (1.f - i / 8.f);
        gx.Line(cx + ca * r * 0.45f, cy + sa * r * 0.45f, cx + ca * r, cy + sa * r, Alpha(c, alpha), w);
    }
    if (moving) SetTimer(v.hwnd, 1, (UINT)(90 - now % 90), nullptr);  // the next spoke (WM_TIMER 1 asks for a frame); nothing drawn between
}

namespace {
// The small copies the wipe draws nearest-neighbour: per image, size and focus, made once (UI thread). src: the art it was made
// from, not kept alive here (a banner's hero stays decoded only as long as the art cache keeps it); a copy of art that is
// gone goes too.
struct Small { std::weak_ptr<const Image> src; ImageP img; };
std::map<std::tuple<const Image*, int, int, int, int>, Small> smalls;
}  // namespace

// The art's part a w x h rectangle shows (as ImageCover crops it: centred across, `focusY` down; as HeroCover's `keep` crops it:
// never below the top `keep` of its height), box-filtered down to sw x sh pixels (premultiplied BGRA averaged).
static ImageP SmallCopy(const ImageP& src, float w, float h, int sw, int sh, float focusY, float keep) {
    keep = std::max(0.5f, std::min(1.f, keep));
    const auto key = std::make_tuple(src.get(), sw, sh, (int)std::lround(focusY * 100), (int)std::lround(keep * 1000));
    auto it = smalls.find(key);
    if (it != smalls.end()) {
        if (!it->second.src.owner_before(src) && !src.owner_before(it->second.src)) return it->second.img;
        smalls.erase(it);  // another image at a gone one's address
    }
    for (auto k = smalls.begin(); k != smalls.end();) k = k->second.src.expired() ? smalls.erase(k) : std::next(k);
    if (smalls.size() > 256) smalls.clear();  // a big library seen at many sizes: start over (each is made again on demand)
    auto out = std::make_shared<Image>();
    out->w = sw; out->h = sh; out->px.assign((size_t)sw * sh * 4, 0);
    const float k = std::max(w / src->w, h / (src->h * keep));
    const float cw = w / k, ch = h / k, cx = (src->w - cw) / 2, cy = std::min((src->h - ch) * focusY, src->h * keep - ch);
    for (int j = 0; j < sh; ++j) {
        const int y0 = std::max(0, (int)std::floor(cy + ch * j / sh)), y1 = std::min(src->h, std::max(y0 + 1, (int)std::floor(cy + ch * (j + 1) / sh)));
        for (int i = 0; i < sw; ++i) {
            const int x0 = std::max(0, (int)std::floor(cx + cw * i / sw)), x1 = std::min(src->w, std::max(x0 + 1, (int)std::floor(cx + cw * (i + 1) / sw)));
            uint32_t sum[4]{}, n = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x) {
                    const uint8_t* p = &src->px[((size_t)y * src->w + x) * 4];
                    for (int c = 0; c < 4; ++c) sum[c] += p[c];
                    ++n;
                }
            uint8_t* d = &out->px[((size_t)j * sw + i) * 4];
            for (int c = 0; c < 4; ++c) d[c] = (uint8_t)(n ? (sum[c] + n / 2) / n : 0);
        }
    }
    smalls[key] = Small{src, out};
    return out;
}

// "960p" from a last run's "1706x960"; empty when it is not known
static wstring Height(const wstring& size) {
    const size_t x = size.find_last_of(L"x×");
    const int h = x == wstring::npos ? 0 : _wtoi(size.c_str() + x + 1);
    return h > 0 ? std::to_wstring(h) + L"p" : wstring();
}

int WipeArtAt(float x, float y) {
    if (v.menu.open || v.dialog.open || v.first.open || x < 0) return 0;
    int owner = 0;
    for (size_t i = v.hits.size(); i-- > 0;) if (v.hits[i].shown && v.hits[i].r.has(x, y)) { owner = v.hits[i].id; break; }
    for (const auto& a : v.wipeArts) if (a.r.has(x, y) && a.owner == owner) return a.id;
    return 0;
}

float UpscaleWipe(Gfx& gx, int id, int owner, const Rc& r, float rad, const ImageP& img, float focusY, const LastRun& last, bool focused, float keep, bool marks) {
    if (cv.dry || gx.dry) return 0;
    if (!img) { v.wipeAt.erase(id); v.wipeOut.erase(id); return 0; }
    // the art in the window (clipped to the part of it the page shows); the pointer on it, on its hit (nothing over it)
    Rc w = r.off(0, -cv.scroll);
    if (cv.view.w > 0) w = Intersect(w, cv.view);
    if (w.w > 0.5f && w.h > 0.5f) v.wipeArts.push_back({id, owner, w});
    // (the hit under the pointer is the last frame's: this frame's hits are still being made)
    const bool pointer = !v.menu.open && !v.dialog.open && !v.first.open && w.has(v.mouseX, v.mouseY) && v.hot == owner;
    if (pointer) v.wipeHot = id;
    float t = 0;
    if (!pointer && !focused) {
        // left: it fades out as it came in (from where its fade in had got to), at once with Animation effects off
        auto at = v.wipeAt.find(id);
        if (at != v.wipeAt.end()) {
            const float shown = Entrance(at->second, kWipeMs);
            v.wipeAt.erase(at);
            if (shown > 0 && v.live && v.hwnd && MotionOn()) {
                v.wipeOut[id] = NowMs() - (ULONGLONG)((1 - shown) * kWipeMs);
                Invalidate();
            }
        }
        auto out = v.wipeOut.find(id);
        if (out == v.wipeOut.end()) return 0;
        t = 1 - Entrance(out->second, kWipeMs);
        if (t <= 0) { v.wipeOut.erase(out); return 0; }
    } else {
        auto at = v.wipeAt.find(id);
        if (at == v.wipeAt.end()) {
            // back before its fade out ended: it comes in from where that had got to
            auto out = v.wipeOut.find(id);
            const float from = out == v.wipeOut.end() ? 0.f : 1 - Entrance(out->second, kWipeMs);
            if (out != v.wipeOut.end()) v.wipeOut.erase(out);
            at = v.wipeAt.emplace(id, NowMs() - (ULONGLONG)(std::max(0.f, from) * kWipeMs)).first;
            if (v.live && v.hwnd && MotionOn()) Invalidate();  // the window's frames carry the fade (MotionPending)
        }
        t = Entrance(at->second, kWipeMs);
    }
    if (t <= 0) return 0;
    // the left half: the render size, pixel-stepped (about a 3.5th of the art's size, nearest-neighbour)
    const int sw = std::max(4, (int)std::lround(r.w / 3.5f)), sh = std::max(4, (int)std::lround(r.h / 3.5f));
    const ImageP pixels = SmallCopy(img, r.w, r.h, sw, sh, focusY, keep);
    const float s = std::max(1.f, gx.scale);
    const float mid = std::round(r.cx() * s) / s;
    gx.PushClip(Rc{r.x, r.y, mid - r.x, r.h});
    gx.ImageNearest(pixels, r, rad, t);
    gx.PopClip();
    if (marks) WipeMarks(gx, r, t, last);
    return t;
}
bool WipeSizes(const LastRun& last) { return !Height(last.renderSize).empty() && !Height(last.outputSize).empty(); }
void WipeMarks(Gfx& gx, const Rc& r, float t, const LastRun& last) {
    if (t <= 0 || cv.dry || gx.dry) return;
    const float s = std::max(1.f, gx.scale);
    const float mid = std::round(r.cx() * s) / s, lw = std::max(2.f, std::round(2 * s)) / s;
    // the split: a 2 DIP white line with a soft shadow (0 0 6px rgba(0,0,0,.45)), kept inside the art
    gx.PushClip(r);
    gx.Shadow(Rc{mid - lw / 2, r.y - 8, lw, r.h + 16}, 0, 6, 0.6f * t);
    gx.Fill(Rc{mid - lw / 2, r.y, lw, r.h}, Alpha(pal.white, t));
    gx.PopClip();
    // the capsules: the input height bottom left, the output's bottom right (10.5 semibold, 18 high; 10 on the small cover);
    // only the sizes this game's last run recorded (none before it ran with WrongGPU: no number is made up)
    const wstring in = Height(last.renderSize), out = Height(last.outputSize);
    if (in.empty() || out.empty()) return;
    const bool big = r.w >= 120;
    TS cs = Ts(big ? 10.5f : 10.f, 600, Alpha(Hex(0x1d1d1f), t)); cs.valign = 1;
    const float pad = big ? 6.f : 5.f, edge = big ? 6.f : 4.f, bottom = big ? 8.f : 6.f;
    const float iw = std::ceil(gx.TextW(in, cs) + 2 * pad), ow = std::ceil(gx.TextW(out, cs) + 2 * pad);
    const Rc ir{std::round(r.x + edge), std::round(r.b() - bottom - 18), iw, 18}, orc{std::round(r.r() - edge - ow), ir.y, ow, 18};
    gx.FillRound(ir, 9, Alpha(pal.white, 0.9f * t));
    gx.FillRound(orc, 9, Alpha(pal.white, 0.9f * t));
    gx.Text(in, Rc{ir.x + pad, ir.y, iw, ir.h}, cs);
    gx.Text(out, Rc{orc.x + pad, orc.y, ow, orc.h}, cs);
}

wstring StatusLine() {
    // in Settings' words (Model data: "Not converted yet" / "Not ready"): none converted, or data that can't be used
    const bool absent = v.modelWhy.empty() || IEq(v.modelWhy, L"no converted model on this PC");
    wstring s = v.modelReady ? L"Model M ready" : absent ? L"Model M not converted" : L"Model M not ready";
    if (!v.executorLine.empty()) s += L" · " + v.executorLine;
    const wstring rr = win::RrBackgroundStatus();
    if (!rr.empty()) s += L" · " + rr;
    return s;
}
bool PlayBlocked(const GameEntry& e) {
    if (e.status == GameStatus::Running) return true;
    // Update all still preparing its games (a read of seconds each): every game it will update waits from the start, not only
    // those queued so far (one started meanwhile would be refused when its turn came, stopping the rest)
    for (const auto& d : v.queueComing) if (IEq(d, e.dir)) return true;
    if (!v.busy || v.task == T_COLLECT) return false;  // collecting logs only reads
    if (v.task != T_NONE || v.busyDir.empty() || IEq(v.busyDir, e.dir)) return true;
    for (const auto& q : v.queue) if (IEq(q.dir, e.dir)) return true;  // still to come in Update all (one it has done is free)
    return false;
}
wstring BusyWhy(const GameEntry& e) {
    if (e.status == GameStatus::Running) return e.name + L" is running. Its actions wait until it closes.";
    // the last session's list (ui_library.cpp): its actions wait for a scan; one that failed asks for Rescan
    if (!v.busy && v.listCached) return v.scanning ? L"Waits until WrongGPU finishes looking for games." : L"Rescan first: WrongGPU couldn't read the games yet.";
    if (!v.busy) return L"";
    switch (v.task) {
        case T_MODEL: return L"Waits until the Model M data is converted.";
        case T_RRDATA: return L"Waits until the Ray Reconstruction data is converted.";
        case T_SELFINSTALL: return L"Waits until WrongGPU is installed for this user.";
        case T_FIRSTRUN: return v.first.installOffer && v.first.choice == 1 ? L"Waits until WrongGPU is installed for this user." : L"Waits until WrongGPU is set up.";
        case T_SELFUNINSTALL: return L"Waits until the launcher is uninstalled.";
        case T_COLLECT: return L"Waits until the logs are collected.";
        default: break;
    }
    wstring who;
    for (const auto& x : v.games) if (IEq(x.dir, v.busyDir)) who = x.name;
    if (who.empty() && IEq(v.cur.dir, v.busyDir)) who = v.cur.name;
    wstring verb = v.busyVerb;
    while (!verb.empty() && (verb.back() == L'…' || verb.back() == L'.')) verb.pop_back();
    if (!verb.empty()) verb[0] = towlower(verb[0]);
    // WrongGPU is the one at work, not the game (the game updating could be its store's update)
    if (LookingAtGames()) return L"Waits until WrongGPU finishes looking at your games.";  // the launcher's uninstall, before it asks
    if (who.empty() || verb.empty() || verb == L"working") return L"Waits until WrongGPU finishes the current task.";
    if (verb == L"installing") return L"Waits until WrongGPU is installed in " + who + L".";
    if (verb == L"uninstalling") return L"Waits until WrongGPU is uninstalled from " + who + L".";
    if (verb == L"removing files") return L"Waits until WrongGPU's files are removed from " + who + L"'s old folder.";  // a game that is gone
    if (verb == L"applying settings") return L"Waits until WrongGPU finishes applying settings to " + who + L".";
    if (verb == L"resetting") return L"Waits until WrongGPU finishes resetting OptiScaler settings in " + who + L".";
    return L"Waits until WrongGPU finishes " + verb + L" " + who + L".";  // "Waits until WrongGPU finishes updating Quarry Nine."
}
bool ToolbarHit(int id) { return id == ID_NAV_LIBRARY || id == ID_NAV_SETTINGS || id == ID_STATUSLINE || id == ID_BRAND || (id >= ID_CAP_MIN && id <= ID_CAP_CLOSE); }

float CaptionW() { return 3 * kCapW; }
int CaptionAt(float x, float y) {
    if (y < 0 || y >= kBarH || x >= v.W || x < v.W - CaptionW()) return 0;
    return ID_CAP_MIN + std::min(2, (int)((x - (v.W - CaptionW())) / kCapW));
}
// Minimize, Maximize / Restore and Close as Windows 11 draws them (Segoe Fluent Icons' chrome glyphs, 10 DIPs; a quiet fill
// under the pointer, Close red with a white glyph), the toolbar's full height. Drawn last, over a sheet's scrim too: the
// window can always be minimized, moved or closed. Not in the Tab order (Alt+Space opens the window menu, as everywhere).
void DrawCaption(Gfx& gx) {
    static const wchar_t glyphs[] = {0xE921, 0xE922, 0xE8BB};
    const wchar_t* names[] = {L"Minimize", v.maximized ? L"Restore" : L"Maximize", L"Close"};
    for (int i = 0; i < 3; ++i) {
        const int id = ID_CAP_MIN + i;
        const Rc r{v.W - CaptionW() + i * kCapW, 0, kCapW, kBarH};
        const bool hot = Hot(id), down = v.down == id && hot;
        Col fg = v.active ? pal.text : pal.text3;
        if (pal.hc && (hot || down)) {  // a Contrast theme: the highlight under the pointer, Close too
            gx.Fill(r, pal.tint);
            fg = pal.onAccent;
        } else if (i == 2 && (hot || down)) {
            gx.Fill(r, down ? Mix(Hex(0xc42b1c), pal.black, 0.12f) : Hex(0xc42b1c));  // Windows' own close red (white on it 5.7:1)
            fg = pal.white;
        } else if (down) gx.Fill(r, Alpha(pal.text, pal.dark ? 0.07f : 0.09f));
        else if (hot) gx.Fill(r, Alpha(pal.text, pal.dark ? 0.1f : 0.06f));
        if (gx.iconFont) gx.Icon(i == 1 && v.maximized ? (wchar_t)0xE923 : glyphs[i], r.cx(), r.cy(), 10, fg);
        else {  // no icon font: drawn lines
            const float cx = std::round(r.cx()), cy = std::round(r.cy());
            if (i == 0) gx.Line(cx - 5, cy + 0.5f, cx + 5, cy + 0.5f, fg);
            else if (i == 1) gx.StrokeRound(Rc{cx - 5, cy - 5, 10, 10}, 1, fg);
            else { gx.Line(cx - 5, cy - 5, cx + 5, cy + 5, fg); gx.Line(cx + 5, cy - 5, cx - 5, cy + 5, fg); }
        }
        AddHit(id, r, R_BUTTON, names[i], true, -1, L"", false);
    }
}

void DrawToolbar(Gfx& gx) {
    gx.Fill(Rc{0, 0, v.W, kBarH}, pal.bg);
    // a hairline under it only while the page scrolls beneath it, fading in over the first 8 DIPs of the scroll (as macOS's
    // and WinUI's bars do), not popping in on the glide's first frame
    if (v.scroll[v.page] > 0.5f && !(v.page == P_GAME && v.pinH > 0)) {
        const float s = std::max(1.f, gx.scale), k = std::min(1.f, v.scroll[v.page] / 8);
        gx.Fill(Rc{0, std::round((kBarH - HairlineW(gx)) * s) / s, v.W, HairlineW(gx)}, Alpha(pal.sep, k));  // inside the bar
    }
    // the brand: WrongGPU, or RightGPU on the page of a game where it is installed and healthy (BrandNow); UI Automation reads
    // the words shown (BE-wide notes: the wordmark's name follows its text; voice access says what it sees)
    float fill = 0, cross = 0;
    const bool flipping = BrandNow(fill, cross);
    if (flipping && v.live) v.barMoving = true;  // the flip's frames come at the display's rate (Animating)
    else if (!flipping && v.flipT < 0) v.flipDir.clear();  // played (or not this page's): later visits start as RightGPU
    AppMark(gx, Rc{20, 14, 24, 24}, fill, cross);
    TS bs = Ts(14, 600, pal.text); bs.valign = 1; bs.spacing = 0.1f;
    const float brandR = 54 + std::max(gx.TextW(L"WrongGPU", bs), cross > 0 ? gx.TextW(L"RightGPU", bs) : 0.f) + 2;
    if (cross < 1) { TS w = bs; w.color = Alpha(bs.color, 1 - cross); gx.Text(L"WrongGPU", Rc{54, 0, brandR - 54, kBarH}, w); }
    if (cross > 0) { TS r = bs; r.color = Alpha(bs.color, cross); gx.Text(L"RightGPU", Rc{54, 0, brandR - 54, kBarH}, r); }
    AddHit(ID_BRAND, Rc{20, 14, brandR - 20, 24}, R_TEXT, cross >= 0.5f ? L"RightGPU" : L"WrongGPU", true, -1, L"", false);
    // Library | Settings: 30 DIP segments in a well of the fill (2 round them), centred; the Library's carries the count of
    // the games that need attention: in red while one of them is broken (a repair, leftovers, a game gone: red is what needs
    // you), accentText 6.3:1 on the raised segment (light), in dark a lighter red, 4.8:1 on the raised segment, 7.2 on the
    // well; in text2 (grey) while they only have an update ready, which is routine (UI review #3)
    struct Nav { int id; const wchar_t* text; bool on; int count; };
    const int attention = NeedsAttentionCount();
    const Nav nav[] = {{ID_NAV_LIBRARY, L"Library", v.page == P_LIBRARY || v.page == P_GAME, attention}, {ID_NAV_SETTINGS, L"Settings", v.page == P_SETTINGS, 0}};
    TS cs = Ts(12, 600, pal.hc ? pal.text : ProblemCount() == 0 ? pal.text2 : pal.dark ? Hex(0xffaaa1) : pal.accentText); cs.valign = 1;
    float ws[2], total = 4;
    for (int i = 0; i < 2; ++i) {
        float w = 32 + gx.TextW(nav[i].text, Ts(13, 600, pal.text));
        if (nav[i].count > 0) w += 6 + gx.TextW(std::to_wstring(nav[i].count), cs);
        ws[i] = std::ceil(std::max(96.f, w));
        total += ws[i] + (i ? 2.f : 0.f);
    }
    // centred at every width and whatever the status line says (it changes as the app runs: the primary nav never moves)
    const Rc well{std::round((v.W - total) / 2), 9, total, 34};
    const wstring status = StatusLine();
    TS ss = Ts(12, 400, pal.text2); ss.valign = 1;
    std::vector<wstring> parts;
    for (size_t a = 0;;) {
        const size_t b = status.find(L" \u00b7 ", a);
        parts.push_back(status.substr(a, b == wstring::npos ? wstring::npos : b - a));
        if (b == wstring::npos) break;
        a = b + 3;
    }
    const float right = v.W - CaptionW() - 14;  // clear of the caption buttons
    gx.FillRound(well, 9, pal.fill);
    HcEdge(gx, well, 9);
    float x = well.x + 2;
    for (int i = 0; i < 2; ++i) {
        const Nav& n = nav[i];
        const Rc r{x, well.y + 2, ws[i], 30};
        const bool hot = Hot(n.id), down = Down(n.id);
        if (pal.hc) {  // a Contrast theme: the page's tab on the highlight, the pointer's an edge
            if (n.on) gx.FillRound(r, 7, pal.tint);
            else if (hot || down) HcEdge(gx, r, 7, true);
        } else if (n.on) {
            gx.FillRound(r.off(0, 1), 7, Alpha(pal.black, 0.09f));  // 0 1px 2px rgba(0,0,0,.12)
            gx.FillRound(r, 7, down ? Mix(pal.segOn, pal.fillHover, 0.5f) : pal.segOn);
        } else if (down) gx.FillRound(r, 7, Mix(pal.fillHover, pal.text, 0.06f));
        else if (hot) gx.FillRound(r, 7, pal.fillHover);
        TS ts = Ts(13, 600, pal.hc && n.on ? pal.onAccent : n.on || hot ? pal.text : pal.text2); ts.valign = 1;
        const wstring cnt = n.count > 0 ? std::to_wstring(n.count) : L"";
        TS ns = cs;
        if (pal.hc && n.on) ns.color = pal.onAccent;
        const float tw = gx.TextW(n.text, ts), cw = cnt.empty() ? 0 : 6 + gx.TextW(cnt, ns);
        const float tx = std::round(r.x + (r.w - tw - cw) / 2);
        gx.Text(n.text, Rc{tx, r.y, tw + 2, r.h}, ts);
        if (!cnt.empty()) gx.Text(cnt, Rc{tx + tw + 6, r.y, cw, r.h}, ns);
        FocusRing(gx, r, 7, n.id);
        AddHit(n.id, r, R_ITEM, wstring(n.text) + (n.count == 1 ? L" (1 game needs attention)" : n.count > 1 ? L" (" + std::to_wstring(n.count) + L" games need attention)" : L""),
               true, n.on ? 1 : 0);
        x += ws[i] + 2;
    }
    // the status line: a dot and one quiet line, right-aligned, its parts apart by a middle dot with 9 DIPs either side (the
    // mockup's spacing) at every width: a narrow window leaves out its last parts whole ("Model M ready · RX 9070 XT"), and
    // then says its first part without "Model M" ("Not converted", the words of Settings' Model data row; the dot keeps its
    // colour), and only what still does not fit is cut with an ellipsis. Anything left out: the whole line in the tooltip
    // (WM_NCHITTEST lets the pointer have it); UI Automation always has the whole line.
    const float room = right - (well.r() + 24) - 15;
    if (room >= 60) {
        const float dotW = gx.TextW(L"\u00b7", ss), gap = 9;
        auto width = [&](size_t n) { float w = 0; for (size_t i = 0; i < n; ++i) w += gx.TextW(parts[i], ss) + 1 + (i ? 2 * gap + dotW : 0.f); return w; };
        size_t n = parts.size();
        while (n > 1 && width(n) > room) --n;
        bool cut = n < parts.size();
        if (width(1) > room && parts[0].rfind(L"Model M ", 0) == 0) {
            parts[0] = parts[0].substr(8);
            parts[0][0] = towupper(parts[0][0]);
            cut = true;
        }
        cut |= width(n) > room;
        const float tw = std::min(room, width(n));
        const Rc tr{right - tw, 0, tw, kBarH};
        StatusDot(gx, tr.x - 11.5f, kBarH / 2, v.modelReady && !v.gpuName.empty() ? pal.green : pal.amber);
        float px = tr.x;
        for (size_t i = 0; i < n; ++i) {
            if (i) { gx.Text(L"\u00b7", Rc{px + gap, 0, dotW + 1, kBarH}, ss); px += 2 * gap + dotW; }
            const float pw = std::min(gx.TextW(parts[i], ss) + 1, tr.r() - px);
            gx.Text(parts[i], Rc{px, 0, pw + 1, kBarH}, ss);
            px += pw;
        }
        AddHit(ID_STATUSLINE, Rc{tr.x - 15, 16, tw + 15, 20}, R_TEXT, status, true, -1, L"", false);
        if (cut) v.tips[ID_STATUSLINE] = status;
    }
}

// ---- overlays ---------------------------------------------------------------------------------------------------------------
// The game page's More menu as it is now: MoreMenu's items and, with WrongGPU's OptiScaler setup, its reset
std::vector<MenuItem> PageMoreMenu() {
    return MoreMenu();  // "Reset OptiScaler settings..." is one of its items (off with its reason in the menu's note, as the others)
}
// The page read again for the same game (an operation's end, a running game's poll, Check again) with its More or Output
// filter menu open: the menu stays under the pointer, its items' states (enabled, checked, words) as they are now. One whose
// items changed (an item came or went: Roll back once an update kept a version) closes, as before; any other menu closes.
void RefreshPageMenu() {
    if (!v.menu.open) return;
    std::vector<MenuItem> now;
    if (v.page == P_GAME && v.menu.owner == ID_G_MORE) now = PageMoreMenu();
    else if (v.page == P_GAME && v.menu.owner == ID_G_FILTER) now = FilterMenu();
    bool same = !now.empty() && now.size() == v.menu.items.size();
    for (size_t i = 0; same && i < now.size(); ++i) same = now[i].id == v.menu.items[i].id && now[i].sep == v.menu.items[i].sep;
    if (!same) { v.menu.open = false; return; }
    for (size_t i = 0; i < now.size(); ++i) {
        MenuItem& it = v.menu.items[i];
        it.text = now[i].text; it.enabled = now[i].enabled; it.checked = now[i].checked; it.help = now[i].help;
    }
    if (v.menu.hot >= 0 && (size_t)v.menu.hot < v.menu.items.size() && !v.menu.items[(size_t)v.menu.hot].enabled) v.menu.hot = -1;
}
void OpenMenu(int owner, const Rc& anchor, std::vector<MenuItem> items, bool right, float width) {
    v.menu.open = true; v.menu.owner = owner; v.menu.anchor = anchor; v.menu.items = std::move(items); v.menu.right = right; v.menu.width = width;
    v.menu.hot = -1;
    v.menu.scroll = -1;
    v.menu.at = NowMs();
    // a menu can open over its control (one that fits neither under nor over it): the second click of the double click that
    // opened it does not choose the item now under the pointer
    win::clickGuardAt = GetTickCount64();
    // it has the keys, its control the focus: opened from UI Automation while a text field had them, Escape and the arrows
    // would stay in the field and the menu could not be closed from the keyboard
    v.focus = owner; v.editFocusId = 0;
    if (v.live && v.hwnd && GetFocus() != v.hwnd) SetFocus(v.hwnd);
    for (size_t i = 0; i < v.menu.items.size(); ++i) if (v.menu.items[i].checked) v.menu.hot = (int)i;
    if (v.focusVisible) for (size_t i = 0; i < v.menu.items.size() && v.menu.hot < 0; ++i) if (!v.menu.items[i].sep && v.menu.items[i].enabled) v.menu.hot = (int)i;
    Invalidate();
}

// A menu's items: 32 DIPs, 28 in a long list (a select's choices; never less: every control is 28 DIPs or more).
float MenuItemH() {
    int n = 0;
    for (auto& it : v.menu.items) n += !it.sep;
    return n > 10 ? 28.f : 32.f;
}
// A note at a menu's end: a separator with words (MenuItem sep + help: why the items that are off are off), its height as
// DrawMenu last measured it (before it places the menu, so every frame's layout has it)
static float menuNoteH = 0;
static bool IsNote(const MenuItem& it) { return it.sep && !it.help.empty(); }
// An item's height: a separator 9, an item with a line under its name (MenuItem::help) 14 more, a note 9 and its words
static float ItemH(const MenuItem& it, float ih) { return it.sep ? (IsNote(it) ? 9.f + menuNoteH : 9.f) : it.help.empty() ? ih : ih + 14; }
static float MenuFullH() {
    const float ih = MenuItemH();
    float h = 10;
    for (auto& it : v.menu.items) h += ItemH(it, ih);
    return h;
}
// The room a menu has: under the toolbar (and under the game page's pinned header when its control is below it, so the
// header's actions stay in view), 8 DIPs from the edges.
static float MenuTop() { return (v.page == P_GAME && v.pinH > 0 && v.menu.anchor.y >= v.pinH - 1 ? v.pinH : kBarH) + 8; }
static float MenuBottom() { return v.H - 8; }
// Where an item's middle is from the top of the menu's whole list (no scroll)
static float MenuItemMid(int index) {
    const float ih = MenuItemH();
    float off = 5;
    for (int i = 0; i < (int)v.menu.items.size(); ++i) {
        const float h = ItemH(v.menu.items[(size_t)i], ih);
        if (i == index) return off + h / 2;
        off += h;
    }
    return off;
}

// Under its control, else over it; a list that fits neither way (the menu key's 21 keys in a short window) takes the room
// there is, centred on its control, and scrolls (the chosen item over the control when it opens, as a pop-up button has it).
Rc MenuRect() {
    const Menu& m = v.menu;
    const float full = MenuFullH(), top = MenuTop(), bottom = MenuBottom();
    float x = m.right ? m.anchor.r() - m.width : m.anchor.x;
    float y, h = full;
    if (m.anchor.b() + 6 + full <= bottom) y = m.anchor.b() + 6;
    else if (m.anchor.y - 6 - full >= top) y = m.anchor.y - 6 - full;
    else if (full <= bottom - top) {  // fits the window but neither under nor over its control: whole, moved as little as it
        // takes (its first or chosen item level with the control, as a pop-up button's), never cut into a scroll with a short
        // note sliced at its edge (the More menu's waiting note at the minimum window)
        int chosen = 0;
        for (size_t i = 0; i < m.items.size(); ++i) if (m.items[i].checked) { chosen = (int)i; break; }
        y = std::min(std::max(top, m.anchor.cy() - MenuItemMid(chosen)), bottom - full);
    } else {  // the chosen item over the control: the list runs up and down from it as far as the room goes (at least 8 items)
        int chosen = 0;
        for (size_t i = 0; i < m.items.size(); ++i) if (m.items[i].checked) { chosen = (int)i; break; }
        const float y0 = m.anchor.cy() - MenuItemMid(chosen), minH = std::min({full, bottom - top, 8 * MenuItemH() + 10});
        y = std::max(top, y0);
        float yb = std::min(bottom, y0 + full);
        if (yb - y < minH) { if (y <= top) yb = std::min(bottom, y + minH); else y = std::max(top, yb - minH); }
        h = yb - y;
    }
    y = std::min(std::max(top, y), std::max(top, bottom - h));  // always in the room (an anchor scrolled out of view too)
    x = std::max(8.f, std::min(x, v.W - m.width - 8));
    return Rc{x, y, m.width, h};
}
float MenuMaxScroll() { return std::max(0.f, MenuFullH() - MenuRect().h); }
// The item `index` brought inside the menu's visible part (the keyboard's choice)
// (clear of the scroll arrows' rows: kMenuArrowH, where items are not drawn)
constexpr float kMenuArrowH = 18;
void MenuShowItem(int index) {
    const float mid = MenuItemMid(index), viewH = MenuRect().h;
    const float ih = index >= 0 && index < (int)v.menu.items.size() ? ItemH(v.menu.items[(size_t)index], MenuItemH()) : MenuItemH();
    float& s = v.menu.scroll;
    if (s < 0) return;  // not placed yet: DrawMenu brings the chosen item into view
    if (mid - ih / 2 - kMenuArrowH < s) s = mid - ih / 2 - kMenuArrowH;
    else if (mid + ih / 2 + kMenuArrowH > s + viewH) s = mid + ih / 2 + kMenuArrowH - viewH;
    s = std::min(std::max(0.f, s), MenuMaxScroll());
}
bool MenuScroll(float dy) {
    if (!v.menu.open || MenuMaxScroll() <= 0) return false;
    if (v.menu.scroll < 0) v.menu.scroll = 0;
    v.menu.scroll = std::min(std::max(0.f, v.menu.scroll + dy), MenuMaxScroll());
    Invalidate();
    return true;
}

void DrawMenu(Gfx& gx) {
    if (!v.menu.open) return;
    Menu& m = v.menu;
    const float ih = MenuItemH();
    const TS noteS = [] { TS s = Ts(12, 400, pal.text2); s.wrap = true; s.lineH = 16; return s; }();
    wstring note;
    for (auto& it : m.items) if (IsNote(it)) { note = it.help; menuNoteH = gx.TextH(note, m.width - 10 - 20, noteS) + 13; }
    const Rc r = MenuRect();
    const float x = r.x, maxScroll = MenuMaxScroll();
    if (m.scroll < 0) {  // placed: the chosen item (else the first) level with its control, or as near as the list allows
        int at = -1;
        for (size_t i = 0; i < m.items.size() && at < 0; ++i) if (m.items[i].checked) at = (int)i;
        m.scroll = at < 0 ? 0.f : MenuItemMid(at) - (std::min(std::max(m.anchor.cy(), r.y), r.b()) - r.y);
        m.scroll = std::min(std::max(0.f, m.scroll), maxScroll);
        if (at >= 0) MenuShowItem(at);  // whole, clear of the scroll arrows' rows
    }
    m.scroll = std::min(std::max(0.f, m.scroll), maxScroll);
    // the entrance: it fades in and settles 4 DIPs toward its control (from above when it opens below it)
    const float e = Entrance(m.at, kMenuMs);
    if (e < 1) { gx.PushFade(e); gx.SetOffset(0, (r.y >= m.anchor.b() ? -4.f : 4.f) * (1 - e)); }
    Sheet(gx, r, 10, 1);
    {  // the sheet for UI Automation (a Menu named after its control), before its items: they take the clicks
        wstring owner;
        for (auto& h : v.hits) if (h.id == m.owner) owner = h.name;
        AddHit(ID_MENUBOX, r, R_MENU, owner, true, -1, L"", false);
    }
    // a list that scrolls: its items clipped to the sheet (5 DIPs in), their hits too (an item out of view is not clicked); with
    // more above or below, that edge's row is the scroll arrow's own (kMenuArrowH): no item's words are drawn under it
    const bool moreUp = maxScroll > 0 && m.scroll > 0.5f, moreDown = maxScroll > 0 && m.scroll < maxScroll - 0.5f;
    const float clipT = r.y + (moreUp ? kMenuArrowH : 5.f), clipB = r.b() - (moreDown ? kMenuArrowH : 5.f);
    const Rc inner{r.x, clipT, r.w, clipB - clipT};
    const ContentView keep = cv;
    if (maxScroll > 0) { gx.PushClip(inner); cv.view = inner; cv.scroll = 0; }
    float iy = r.y + 5 - m.scroll;
    for (size_t i = 0; i < m.items.size(); ++i) {
        const MenuItem& it = m.items[i];
        // a list that scrolls shows whole items only: one cut by its edge is left out (no sliced words beside the arrow)
        if (maxScroll > 0 && (iy < clipT - 0.5f || iy + ItemH(it, ih) > clipB + 0.5f)) { iy += ItemH(it, ih); continue; }
        if (it.sep) {
            Hairline(gx, x + 11, iy + 4, x + m.width - 11);
            if (IsNote(it)) {  // the note's words in the secondary colour, read by UI Automation as the menu's text
                const Rc nr{x + 15, iy + 9 + 5, m.width - 10 - 20, menuNoteH - 13 + 2};
                gx.Text(it.help, nr, noteS);
                AddText(nr, it.help);
                iy += menuNoteH;
            }
            iy += 9;
            continue;
        }
        const float itemH = ItemH(it, ih);
        const Rc ir{x + 5, iy, m.width - 10, itemH};
        const int id = ID_MENU + (int)i;
        const bool hot = it.enabled && ((int)i == m.hot || Hot(id));
        if (hot) gx.FillRound(ir, 6, pal.hc ? pal.tint : pal.fill);  // a Contrast theme: the highlight, its words in its text colour
        const bool hcHot = pal.hc && hot;
        const Col c = hcHot ? pal.onAccent : !it.enabled ? pal.text3 : it.danger ? pal.accentText : pal.text;
        if (it.icon) gx.Icon(it.icon, ir.x + 17, ir.cy(), 13, hcHot ? pal.onAccent : !it.enabled ? pal.text3 : it.danger ? pal.accentText : pal.text2);
        TS ts = Ts(13.5f, 400, c); ts.valign = 1;
        const float tx = ir.x + (it.icon ? 36.f : 10.f);
        if (it.help.empty()) gx.Text(it.text, Rc{tx, ir.y, ir.w - 56, ir.h}, ts);
        else {  // the name, and what it does under it in the secondary colour (cut with an ellipsis where it ends)
            gx.Text(it.text, Rc{tx, ir.y + 5, ir.w - 56, 20}, ts);
            TS hs = Ts(12, 400, hcHot ? pal.onAccent : it.enabled ? pal.text2 : pal.text3); hs.valign = 1; hs.trim = true;
            gx.Text(it.help, Rc{tx, ir.y + 24, ir.w - (tx - ir.x) - 30, 16}, hs);
        }
        if (it.checked) gx.Icon(ic::check, ir.r() - 16, ir.cy(), 12, hcHot ? pal.onAccent : pal.tintText);
        AddHit(id, ir, R_MENUITEM, it.text, it.enabled, it.checked ? 1 : -1);  // checked: UI Automation's Toggle "on"
        Describe(id, it.help.empty() && !it.enabled ? note : it.help);  // an item that is off: why, as its help text (the note)
        iy += itemH;
    }
    if (maxScroll > 0) {
        gx.PopClip();
        cv = keep;
        // more above or below: a small chevron in that edge's own row (on the bare sheet, over no words) says which way
        if (moreUp) gx.Icon(ic::up, r.cx(), r.y + kMenuArrowH / 2 + 1, 9, pal.text2);
        if (moreDown) gx.Icon(ic::down, r.cx(), r.b() - kMenuArrowH / 2 - 1, 9, pal.text2);
    }
    if (e < 1) { gx.SetOffset(0, 0); gx.PopFade(); }
}

int DialogDefault(const Dialog& d) {
    int def = -1;
    for (size_t i = 0; i < d.buttons.size(); ++i)
        if (d.buttons.size() == 1 || (d.buttons[i].primary && !d.buttons[i].danger && d.icon != 1 && d.icon != 3)) def = (int)i;  // an error's Close too
    return def;
}

// A dialog: option E's sheet (radius 14) over the blurred, dimmed window. The mark (a question) or a glyph tile (a warning:
// amber; an error: red, accentText 5.4:1 on the tile in light, 5.6:1 in dark; a note: text2) on the left, the title and the
// message beside it, capsule buttons at the bottom right.
void DrawDialog(Gfx& gx) {
    if (!v.dialog.open) return;
    Dialog& d = v.dialog;
    // the entrance: the page blurs and dims while the sheet fades in and settles 8 DIPs
    const float e = Entrance(d.at, kDialogMs);
    Scrim(gx, e);
    const float btnH = 34;
    float row = 48;  // the buttons in one row with the sheet's margins: a sheet wide enough for long labels (never past the window)
    for (auto& b : d.buttons) row += std::max(84.f, ButtonW(gx, b.text, 0, btnH, b.primary ? B_GO : B_SECONDARY)) + 8;
    const float w = std::min(std::max(460.f, std::ceil(row - 8)), v.W - 48), tx = 24 + 36 + 16, tw = w - tx - 24;
    TS bs = Ts(13.5f, 400, pal.text2); bs.wrap = true; bs.lineH = 20;
    const float bh = gx.TextH(d.body, tw, bs);
    TS tsT = Ts(17, 600, pal.text, F_DISPLAY); tsT.wrap = true; tsT.lineH = 22;
    const float th = d.title.empty() ? 0 : gx.TextH(d.title, tw, tsT);
    const float foot = 20 + btnH + 20;
    const float h = std::min(v.H - 48, 24 + std::max(36.f, th + (th > 0 && bh > 0 ? 6 : 0) + bh) + foot);
    const Rc r{std::round((v.W - w) / 2), std::round(std::max(24.f, (v.H - h) / 2)), w, h};
    // a text longer than the room (a failure with its warnings in a short window) scrolls: the wheel or Up / Down
    if (e < 1) { gx.PushFade(e); gx.SetOffset(0, 8 * (1 - e)); }
    Sheet(gx, r, 14, 1.6f);
    // the sheet for UI Automation (a dialog named by its title, else by its kind), and its text, before its buttons
    AddHit(ID_DIALOGBOX, r, R_DIALOG, !d.title.empty() ? d.title : d.icon == 1 || d.icon == 3 ? L"Warning" : d.icon == kIconError ? L"Error" : d.icon == 2 ? L"Note" : L"Question",
           true, -1, L"", false);
    if (!d.title.empty() && !d.body.empty()) Describe(ID_DIALOGBOX, d.body);
    const Rc ib{r.x + 24, r.y + 24, 36, 36};
    if (d.icon == 0) AppMark(gx, ib);
    else {
        gx.FillRound(ib, 9, pal.fill);
        const bool warn = d.icon == 1 || d.icon == 3 || d.icon == kIconError;
        gx.Icon(warn ? ic::warn : ic::info, ib.cx(), ib.cy(), 17, d.icon == kIconError ? pal.accentText : warn ? pal.amberText : pal.text2);
    }
    // a title alone sits level with the mark's middle; with a message under it, it starts at the mark's top
    float ty = r.y + 24 + (bh > 0 ? 0.f : std::max(0.f, (36 - th) / 2));
    if (th > 0) { gx.Text(d.title, Rc{r.x + tx, ty, tw, th + 2}, tsT); ty += th + 6; }
    const float room = std::max(0.f, r.b() - foot + 8 - ty);
    // only a text taller than its room scrolls (a title alone, or a short message, never shows a thumb or a fade)
    d.maxScroll = bh > 0 && bh > room + 0.5f ? std::ceil(bh + 4 - room) : 0.f;
    d.scroll = std::min(std::max(0.f, d.scroll), d.maxScroll);
    gx.PushClip(Rc{r.x + tx, ty, tw, room});
    gx.Text(d.body, Rc{r.x + tx, ty - d.scroll, tw, bh + 4}, bs);
    gx.PopClip();
    AddText(Rc{r.x + tx, ty, tw, std::max(1.f, std::min(room, bh + 4))}, d.body, ID_DIALOGBODY);
    if (d.maxScroll > 0) {  // more above or below: the sheet fades over the cut line, and a thin thumb says where it is
        if (d.scroll > 0.5f) gx.GradStops(Rc{r.x + tx, ty, tw, 18}, 0, {{0.f, pal.sheet}, {1.f, Alpha(pal.sheet, 0)}}, 0, 0, 0, 1);
        if (d.scroll < d.maxScroll - 0.5f) gx.GradStops(Rc{r.x + tx, ty + room - 18, tw, 18}, 0, {{0.f, Alpha(pal.sheet, 0)}, {1.f, pal.sheet}}, 0, 0, 0, 1);
        const float thH = std::max(24.f, room * room / (bh + 4)), thY = ty + (room - thH) * (d.scroll / d.maxScroll);
        gx.FillRound(Rc{r.r() - 9, thY, 4, thH}, 2, Alpha(pal.text2, 0.45f));
    }
    float bx = r.r() - 24;
    const int def = DialogDefault(d);
    for (size_t i = d.buttons.size(); i-- > 0;) {
        const DialogButton& b = d.buttons[i];
        // red only for what needs you; the tint's fill only on the button Return answers (a warning's buttons are equals), and
        // never on a failure's or a warning's lone Close: green says WrongGPU is working (IDENTITY.md), not "this went wrong"
        const bool alarm = d.icon == kIconError || d.icon == 1 || d.icon == 3;
        const BtnKind k = !b.primary ? B_SECONDARY : b.danger || b.attention ? B_PRIMARY : def == (int)i && !alarm ? B_GO : B_SECONDARY;
        bx -= Button(gx, ID_DIALOG + (int)i, bx, r.b() - 20 - btnH, b.text, k, 0, true, btnH, 84, true) + 8;
    }
    if (e < 1) { gx.SetOffset(0, 0); gx.PopFade(); }
}

void ShowToast(const wstring& text, int tone) {
    const ULONGLONG now = NowMs();
    // a toast already up changes its words in place; a new one makes its entrance
    if (v.toast.text.empty() || now > v.toast.until) v.toast.at = now;
    // 7 s, or longer when Windows says so (Accessibility > Visual effects > "Dismiss notifications after this amount of time",
    // SPI_GETMESSAGEDURATION in seconds): it is the only place some results show
    ULONG secs = 0;
    const UINT ms = SystemParametersInfoW(SPI_GETMESSAGEDURATION, 0, &secs, 0) ? (UINT)std::min<ULONG>(std::max<ULONG>(secs, 7), 300) * 1000 : 7000;
    v.toast.text = text; v.toast.tone = tone; v.toast.until = now + ms;
    // the timer wakes the window for the fade out (the animation timer then runs to the end), or takes it away at once
    if (v.hwnd && v.live) SetTimer(v.hwnd, 2, MotionOn() ? (UINT)(ms - kToastOutMs) : ms + 100, nullptr);
    UiaNotify(text, tone == 2, tone == 1 ? 1 : tone == 2 ? 2 : 0);  // it takes no focus: Narrator hears it as an announcement (a result, a failure)
    Invalidate();
}

// A toast: a floating capsule on the blurred material (as the unsaved-changes capsule), a status dot and the words, centred
// low in the window (above that capsule while it shows).
void DrawToast(Gfx& gx) {
    v.toastRc = Rc{};
    if (v.toast.text.empty() || (v.live && NowMs() > v.toast.until)) return;
    const float maxW = std::min(560.f, v.W - 48);
    TS ts = Ts(13.5f, 400, pal.text); ts.wrap = true; ts.lineH = 19;
    const float tw = std::min(maxW - 58, gx.TextW(v.toast.text, ts) + 4), th = gx.TextH(v.toast.text, tw, ts);
    const float w = tw + 58, h = std::max(44.f, th + 24);
    const float bottom = GameBarH() > 0 ? v.H - GameBarH() - 12 : v.H - 24;
    // centred on the page's content (the column, Settings' lists), as the unsaved capsule is, kept in the window
    const float cx = v.pageCx > 0 ? v.pageCx : v.W / 2;
    Rc r{std::round(std::min(std::max(16.f, cx - w / 2), v.W - 16 - w)), std::round(bottom - h), w, h};
    // above the unsaved capsule it would sit on the page's lowest rows, where a row's action can be (the Engine.ini card's file
    // check after Show me how): it goes up until no control of the page is under it, while it stays clear of the header (r5 #9)
    if (GameBarH() > 0)
        for (int pass = 0; pass < 6; ++pass) {
            float top = v.H;  // the highest control under it
            for (auto& hh : v.hits) {
                if (!hh.shown || Passive(hh.role) || hh.role == R_SCROLL || hh.role == R_ITEM) continue;
                const Rc o = Intersect(hh.r, r);
                if (o.w >= 8 && o.h >= 8) top = std::min(top, hh.r.y);  // (a sliver at its edge isn't covering it)
            }
            if (top >= v.H) break;  // nothing under it
            const float y = std::round(top - 12 - h);  // just above that control
            if (y < 120) break;     // no room above: it stays where it was put last
            r.y = y;
        }
    // the entrance: it rises 8 DIPs as it fades in; it fades out over its last quarter second
    const float e = Entrance(v.toast.at, kToastMs);
    float a = e;
    if (v.live && v.motionAt < 0 && MotionOn()) a *= std::min(1.f, (float)((long long)v.toast.until - (long long)NowMs()) / kToastOutMs);
    a = std::max(0.f, a);
    // it takes no clicks, and under the pointer it nearly fades away, so what it covers can be read and clicked
    v.toastRc = r;
    bool over = v.mouseX >= 0 && r.has(v.mouseX, v.mouseY);
    // over the keyboard's focus too (Tab to a control under it): the same, so its ring and words show through
    if (!over && v.focusVisible && v.focus)
        for (auto& f : v.hits)
            if (f.id == v.focus && f.shown) { const Rc o = Intersect(f.r.in(-4), r); over = o.w > 0 && o.h > 0; break; }
    // its words under an open menu (drawn over it): the same, so no word of the toast shows cut at the menu's edge (a menu over
    // its edge alone leaves it as it is)
    if (!over && v.menu.open) { const Rc o = Intersect(MenuRect(), Rc{r.x + 14, r.y + (h - th) / 2, r.w - 28, th}); over = o.w > 0 && o.h > 0; }
    if (over) a *= 0.12f;
    gx.SetOffset(0, 8 * (1 - e));
    Material(gx, r, std::min(h / 2, 22.f), 18, a);
    if (a < 1) gx.PushFade(a);
    StatusDot(gx, r.x + 22, r.y + 12 + 9.5f, v.toast.tone == 1 ? pal.green : v.toast.tone == 2 ? pal.amber : pal.greyDot);
    gx.Text(v.toast.text, Rc{r.x + 36, r.y + (h - th) / 2, tw + 2, th + 2}, ts);
    if (a < 1) gx.PopFade();
    gx.SetOffset(0, 0);
}

// The scroll thumb in the page's area (under the toolbar), between the game page's pinned plate and its unsaved capsule. An
// overlay, as Windows 11 has it: it shows while the page scrolls and while the pointer is over its track (wider then), and
// fades away kThumbHoldMs after (at once with Animation effects off). Offscreen it shows on a page scrolled away from its
// top (a frame "just scrolled"), not at rest. Its hit is the whole track: a press beside the thumb pages (launcher_ui.cpp).
float ThumbAlpha() {
    if (Hot(ID_SCROLLBAR) || v.dragThumb) return 1;
    if (!v.live) return v.scroll[v.page] > 0.5f ? 1.f : 0.f;
    const float idle = (float)std::max(0LL, (long long)NowMs() - (long long)v.scrollSeenAt);
    if (idle < kThumbHoldMs) return 1;
    return MotionOn() ? std::max(0.f, 1 - (idle - kThumbHoldMs) / kThumbFadeMs) : 0.f;
}
void DrawScrollbar(Gfx& gx, Page p) {
    v.thumbRc = Rc{};
    const Rc view = cv.view;
    const float ch = v.contentH[p];
    if (ch <= v.H + 1) return;
    const float top = view.y, visible = v.H - top, total = std::max(visible + 1, ch - top);
    const float t0 = p == P_GAME && v.pinH > 0 ? std::max(0.f, v.pinH - top) : 0, t1 = p == P_GAME ? GameBarH() : 0;
    const float track = visible - t0 - t1;
    const float th = std::max(40.f, track * visible / total);
    const float maxScroll = ch - v.H;
    // a height that just shrank (the page reports its new one as it paints) never puts the thumb under its track
    const float at = std::min(1.f, std::max(0.f, v.scroll[p] / maxScroll));
    const float ty = top + t0 + (track - th) * at;
    const bool hot = Hot(ID_SCROLLBAR) || v.dragThumb;
    static bool wasHot = false;
    static float k = 0, from = 0;  // 0 the resting thumb .. 1 the wide one under the pointer
    if (hot != wasHot) { wasHot = hot; from = k; thumbAt = NowMs(); }
    k = !v.live ? (hot ? 1.f : 0.f) : from + ((hot ? 1.f : 0.f) - from) * Entrance(thumbAt, kThumbGrowMs);  // at once with Animation effects off
    const Rc r{view.r() - 9 - k, ty + 3, 5 + 2 * k, th - 6};
    const float a = ThumbAlpha();
    if (a > 0.01f) gx.FillRound(r, r.w / 2, Alpha(pal.text2, (pal.hc ? 1.f : 0.4f + 0.3f * k) * a));  // a Contrast theme: solid
    v.thumbRc = Rc{view.r() - 14, ty, 14, th};
    const ContentView keep = cv;
    cv.scroll = 0; cv.view = Rc{0, 0, v.W, v.H};
    AddHit(ID_SCROLLBAR, Rc{view.r() - 14, top + t0, 14, track}, R_SCROLL, L"Scroll bar", true, -1, L"", false);
    cv = keep;
}

}  // namespace wg::app::lui
