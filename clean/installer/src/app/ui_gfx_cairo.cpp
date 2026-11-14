// SPDX-License-Identifier: MIT
// WrongGPU launcher window on Linux: the Gfx class (ui_shell.h) drawn with Cairo, Pango and fontconfig instead of Direct2D,
// DirectWrite and WIC (ui_gfx.cpp), so the pages' drawing code is the same on both systems. Offscreen frames (ui-render,
// ui-dump) work; the live window's swap chain has no counterpart here yet (BeginHwnd, Resize: the window layer, slice 1b).
// Fonts: Selawik (text, display) and Fluent System Icons (icons), vendored under assets/fonts and registered with fontconfig
// in Init; the mono face is the system's "monospace". Written for this project.
#include "app/ui_shell.h"

#include <cairo.h>
#include <fontconfig/fontconfig.h>
#include <pango/pangocairo.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <unistd.h>

namespace wg::app::lui {

namespace {
constexpr double kPi = 3.14159265358979323846;

std::string U8(const wstring& w) {  // wchar_t is UTF-32 here
    std::string o;
    o.reserve(w.size());
    for (wchar_t wc : w) {
        const uint32_t c = (uint32_t)wc;
        if (c < 0x80) o += (char)c;
        else if (c < 0x800) { o += (char)(0xC0 | c >> 6); o += (char)(0x80 | (c & 63)); }
        else if (c >= 0xD800 && c < 0xE000) o += "\xEF\xBF\xBD";
        else if (c < 0x10000) { o += (char)(0xE0 | c >> 12); o += (char)(0x80 | ((c >> 6) & 63)); o += (char)(0x80 | (c & 63)); }
        else if (c < 0x110000) { o += (char)(0xF0 | c >> 18); o += (char)(0x80 | ((c >> 12) & 63)); o += (char)(0x80 | ((c >> 6) & 63)); o += (char)(0x80 | (c & 63)); }
    }
    return o;
}

void SetSrc(cairo_t* cr, Col c) { cairo_set_source_rgba(cr, c.r, c.g, c.b, c.a); }

// A rounded rectangle with a radius per corner. Radii that do not fit are scaled down together (as CSS does); one radius is
// never more than half a side, as Direct2D's rounded rectangle clamps it.
void RoundPath(cairo_t* cr, const Rc& r, float tl, float tr, float br, float bl) {
    tl = std::max(0.f, tl); tr = std::max(0.f, tr); br = std::max(0.f, br); bl = std::max(0.f, bl);
    float f = 1;
    if (tl + tr > 0) f = std::min(f, r.w / (tl + tr));
    if (bl + br > 0) f = std::min(f, r.w / (bl + br));
    if (tl + bl > 0) f = std::min(f, r.h / (tl + bl));
    if (tr + br > 0) f = std::min(f, r.h / (tr + br));
    f = std::max(0.f, f);
    tl *= f; tr *= f; br *= f; bl *= f;
    cairo_new_path(cr);
    cairo_move_to(cr, r.x + tl, r.y);
    cairo_line_to(cr, r.r() - tr, r.y);
    if (tr > 0) cairo_arc(cr, r.r() - tr, r.y + tr, tr, -kPi / 2, 0);
    cairo_line_to(cr, r.r(), r.b() - br);
    if (br > 0) cairo_arc(cr, r.r() - br, r.b() - br, br, 0, kPi / 2);
    cairo_line_to(cr, r.x + bl, r.b());
    if (bl > 0) cairo_arc(cr, r.x + bl, r.b() - bl, bl, kPi / 2, kPi);
    cairo_line_to(cr, r.x, r.y + tl);
    if (tl > 0) cairo_arc(cr, r.x + tl, r.y + tl, tl, kPi, kPi * 1.5);
    cairo_close_path(cr);
}

// Three box passes approximate a Gaussian of `sigma` pixels (the usual box widths for a Gaussian); the edge pixel repeats
// past the edge. Premultiplied BGRA, 4 channels of 8 bits, `stride` bytes per row.
void BoxPass(uint8_t* p, int n, int step, int r, std::vector<int>& line) {  // one line of n pixels, `step` bytes apart
    for (int ch = 0; ch < 4; ++ch) {
        const uint8_t* q = p + ch;
        line.assign((size_t)n, 0);
        const int win = 2 * r + 1;
        long sum = 0;
        for (int i = -r; i <= r; ++i) sum += q[(size_t)std::clamp(i, 0, n - 1) * step];
        for (int i = 0; i < n; ++i) {
            line[(size_t)i] = (int)((sum + win / 2) / win);
            sum += q[(size_t)std::min(i + r + 1, n - 1) * step] - q[(size_t)std::max(i - r, 0) * step];
        }
        uint8_t* o = p + ch;
        for (int i = 0; i < n; ++i) o[(size_t)i * step] = (uint8_t)line[(size_t)i];
    }
}
void BoxBlur(uint8_t* px, int w, int h, int stride, float sigma) {
    if (sigma < 0.3f || w < 1 || h < 1) return;
    const float ideal = std::sqrt(12 * sigma * sigma / 3 + 1);
    int wl = (int)std::floor(ideal);
    if (wl % 2 == 0) --wl;
    const int wu = wl + 2;
    const int m = (int)std::lround((12 * sigma * sigma - 3 * wl * wl - 12 * wl - 9) / (-4.f * wl - 4));
    std::vector<int> line;
    for (int pass = 0; pass < 3; ++pass) {
        const int r = ((pass < m ? wl : wu) - 1) / 2;
        if (r < 1) continue;
        for (int y = 0; y < h; ++y) BoxPass(px + (size_t)y * stride, w, 4, r, line);
        for (int x = 0; x < w; ++x) BoxPass(px + (size_t)x * 4, h, stride, r, line);
    }
}

bool DirHas(const std::string& d) { return access((d + "/selawik/selawk.ttf").c_str(), R_OK) == 0; }

// assets/fonts next to the executable (an installed or packaged build), else the build's own define (a development build)
std::string FontDir() {
    char exe[4096];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n > 0) {
        std::string d(exe, (size_t)n);
        d.resize(d.rfind('/') == std::string::npos ? 0 : d.rfind('/'));
        for (const char* rel : {"/assets/fonts", "/../assets/fonts", "/../share/wronggpu/assets/fonts"})
            if (DirHas(d + rel)) return d + rel;
    }
#ifdef WRONGGPU_FONT_DIR
    if (DirHas(WRONGGPU_FONT_DIR)) return WRONGGPU_FONT_DIR;
#endif
    return {};
}

// Segoe Fluent Icons codepoint -> Fluent System Icons codepoint, by the glyph's meaning. Names are the font's own
// (FluentSystemIcons-Regular.json from microsoft/fluentui-system-icons commit 67e6dec1ee956af3ab4c746a2b3bb6a2e5520a44, a
// build input that is not vendored), the _20_regular glyph of each (they all exist at that size).
struct IconPair { wchar_t segoe, fluent; const char* name; };
const IconPair kIcons[] = {
    {ic::grid, 0xF462, "grid_20"},
    {ic::gear, 0xF6A9, "settings_20"},
    {ic::refresh, 0xF13D, "arrow_clockwise_20"},
    {ic::add, 0xF109, "add_20"},
    {ic::download, 0xF150, "arrow_download_20"},
    {ic::search, 0xF68F, "search_20"},
    {ic::back, 0xF2AA, "chevron_left_20"},
    {ic::play, 0xF605, "play_20"},
    {ic::more, 0xE824, "more_horizontal_20"},
    {ic::warn, 0xF869, "warning_20"},
    {ic::folder, 0xF418, "folder_20"},
    {ic::cube, 0xF335, "cube_20"},
    {ic::panel, 0xE8B0, "panel_left_20"},
    {ic::monitor, 0xF359, "desktop_20"},
    {ic::down, 0xF2A3, "chevron_down_20"},
    {ic::check, 0xF294, "checkmark_20"},
    {ic::close, 0xF369, "dismiss_20"},
    {ic::info, 0xF4A3, "info_20"},
    {ic::doc, 0xF378, "document_20"},
    {ic::zip, 0xF435, "folder_zip_20"},
    {ic::wrench, 0xEE86, "wrench_20"},
    {ic::undo, 0xF199, "arrow_undo_20"},
    {ic::trash, 0xF34C, "delete_20"},
    {ic::broom, 0xF201, "broom_20"},
    {ic::upload, 0xF1A4, "arrow_upload_20"},
    {ic::link, 0xF4E4, "link_20"},
    {ic::minus, 0xEBD0, "subtract_20"},
    {ic::shield, 0xF6BE, "shield_20"},
    {ic::question, 0xF63D, "question_circle_20"},
    {ic::chip, 0xE4AC, "developer_board_20"},  // the font has no chip glyph
    {ic::sync, 0xF190, "arrow_sync_20"},
    {ic::key, 0xF4B6, "key_20"},
    {ic::playLine, 0xF605, "play_20"},
    {ic::up, 0xF2B6, "chevron_up_20"},
    {ic::arrowUp, 0xF19B, "arrow_up_20"},
    {ic::palette, 0xF2F5, "color_20"},
    {ic::chevron, 0xF2B0, "chevron_right_20"},
    {ic::external, 0xF582, "open_20"},
    {ic::processor, 0xE4AC, "developer_board_20"},
    {0xE921, 0xEBD0, "subtract_20"},         // caption: minimize (ui_widgets.cpp DrawCaption)
    {0xE922, 0xEB75, "square_20"},           // maximize
    {0xE8BB, 0xF369, "dismiss_20"},          // close
    {0xE923, 0xEB96, "square_multiple_20"},  // restore
    {0xE793, 0xE452, "dark_theme_20"},       // Settings > Appearance (ui_settings.cpp kGlyphAppearance)
    {0xEB3C, 0xF67C, "ruler_20"},            // Settings > Exact mode (kGlyphExact)
};
}  // namespace

// ---- the drawing state --------------------------------------------------------------------------------------------------
struct GfxCairo {
    cairo_surface_t* surf = nullptr;
    cairo_t* cr = nullptr;
    int pw = 0, ph = 0;  // the target's size in pixels
    PangoFontMap* map = nullptr;
    PangoContext* ctx = nullptr;
    std::string fam[4];
    unsigned long long frame = 0;
    // what is open: a clip (cairo_save) or a fade (cairo_push_group)
    struct Layer { bool fade; float opacity; };
    std::vector<Layer> layers;
    // Made once and kept across frames and targets (a Pango layout belongs to the font context, not to the target): text
    // layouts by their text, style and box. Those not used for a while are let go (Gfx::Trim).
    struct CachedLayout {
        PangoLayout* l = nullptr;
        PangoAttrList* base = nullptr;  // the layout's attributes without the lead's colour
        std::string u8;
        float w = 0, h = 0, dx = 0;     // logical size; Balance's shift
        unsigned long long used = 0;
        int leadBytes = -1;             // TextLead's lead length in bytes (-1: not a lead layout)
        Col leadCol{-1, -1, -1, -1};    // the colour set on the lead now
    };
    std::map<std::tuple<wstring, int, int, int, int, int, int>, CachedLayout> layouts;
    // the images' surfaces, made once and shared by every target; shadow: ImageContain's blurred drop shadow
    struct Bmp {
        std::weak_ptr<const Image> img;
        cairo_surface_t* s = nullptr;
        unsigned long long used = 0;
        cairo_surface_t* shadow = nullptr;
        std::tuple<int, int, int> shadowKey{0, 0, 0};
    };
    std::map<const Image*, Bmp> bitmaps;
    cairo_surface_t* frozen = nullptr;
    unsigned long long frozenKey = 0, frozenUsed = 0;
    struct Shadowed { cairo_surface_t* s; unsigned long long used; };
    std::map<std::array<int, 5>, Shadowed> shadows;  // Shadow's blurred masks by shape and strength

    ~GfxCairo() {
        DropTarget();
        for (auto& l : layouts) FreeLayout(l.second);
        for (auto& b : bitmaps) FreeBmp(b.second);
        for (auto& e : shadows) cairo_surface_destroy(e.second.s);
        if (ctx) g_object_unref(ctx);
        if (map) g_object_unref(map);
    }
    static void FreeLayout(CachedLayout& c) {
        if (c.l) g_object_unref(c.l);
        if (c.base) pango_attr_list_unref(c.base);
        c.l = nullptr; c.base = nullptr;
    }
    static void FreeBmp(Bmp& b) {
        if (b.s) cairo_surface_destroy(b.s);
        if (b.shadow) cairo_surface_destroy(b.shadow);
        b.s = b.shadow = nullptr;
    }
    void DropTarget() {
        // a layer left open (an early return in a page) is closed so the target is not left with a pending group
        while (!layers.empty()) PopLayer();
        if (cr) cairo_destroy(cr);
        if (surf) cairo_surface_destroy(surf);
        cr = nullptr; surf = nullptr;
        if (frozen) cairo_surface_destroy(frozen);
        frozen = nullptr; frozenKey = 0;
    }
    void PopLayer() {
        if (layers.empty()) return;
        const Layer l = layers.back();
        layers.pop_back();
        if (!cr) return;
        if (l.fade) {
            cairo_pop_group_to_source(cr);
            cairo_paint_with_alpha(cr, l.opacity);
        } else {
            cairo_restore(cr);
        }
    }
};

Gfx::~Gfx() { delete im_; }

bool Gfx::Init() {
    if (im_) return true;
    auto g = std::make_unique<GfxCairo>();
    FcInit();
    // the vendored fonts join fontconfig's set for this process only (nothing is installed); the font map must be made after
    const std::string dir = FontDir();
    if (!dir.empty()) FcConfigAppFontAddDir(nullptr, (const FcChar8*)dir.c_str());
    else fprintf(stderr, "wronggpu: assets/fonts not found; text falls back to the system's fonts\n");
    g->map = pango_cairo_font_map_new();
    g->ctx = pango_font_map_create_context(g->map);
    // no hinting and no rounded metrics: a layout measures the same at every scale and a glyph sits where DirectWrite's
    // natural layout puts it; grayscale antialiasing (the Windows offscreen frames are grayscale too)
    cairo_font_options_t* fo = cairo_font_options_create();
    cairo_font_options_set_antialias(fo, CAIRO_ANTIALIAS_GRAY);
    cairo_font_options_set_hint_style(fo, CAIRO_HINT_STYLE_NONE);
    cairo_font_options_set_hint_metrics(fo, CAIRO_HINT_METRICS_OFF);
    pango_cairo_context_set_font_options(g->ctx, fo);
    cairo_font_options_destroy(fo);
    pango_cairo_context_set_resolution(g->ctx, 96);
    pango_context_set_round_glyph_positions(g->ctx, FALSE);
    g->fam[F_TEXT] = g->fam[F_DISPLAY] = "Selawik,sans-serif";
    g->fam[F_MONO] = "monospace";
    g->fam[F_ICON] = "FluentSystemIcons-Regular";
    iconFont = false;
    PangoFontFamily** fams = nullptr;
    int nf = 0;
    pango_font_map_list_families(g->map, &fams, &nf);
    for (int i = 0; i < nf; ++i)
        if (!strcmp(pango_font_family_get_name(fams[i]), "FluentSystemIcons-Regular")) iconFont = true;
    g_free(fams);
    im_ = g.release();
    return true;
}

// The live window is slice 1b's: no swap chain here, so there is nothing to begin.
bool Gfx::BeginHwnd(HWND, UINT) { return false; }
void Gfx::Resize(HWND) {}
void Gfx::DropTarget() { if (im_) im_->DropTarget(); }

static void ApplyMatrix(GfxCairo& g, float scale, float ox, float oy) {
    cairo_identity_matrix(g.cr);
    cairo_scale(g.cr, scale, scale);
    cairo_translate(g.cr, ox, oy);
}

bool Gfx::BeginOffscreen(int w, int h, float s) {
    if (!im_) return false;
    GfxCairo& g = *im_;
    ++g.frame;
    Trim();
    g.DropTarget();
    offscreen = true;
    scale = s;
    g.pw = (int)std::lround(w * s);
    g.ph = (int)std::lround(h * s);
    if (g.pw < 1 || g.ph < 1) return false;
    g.surf = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, g.pw, g.ph);  // premultiplied BGRA on little-endian, as WIC's
    if (cairo_surface_status(g.surf) != CAIRO_STATUS_SUCCESS) { g.DropTarget(); return false; }
    g.cr = cairo_create(g.surf);
    pango_cairo_update_context(g.cr, g.ctx);
    ox_ = oy_ = 0;
    ApplyMatrix(g, scale, 0, 0);
    return true;
}

void Gfx::EndFrame() {
    if (!im_ || !im_->cr) return;
    GfxCairo& g = *im_;
    while (!g.layers.empty()) g.PopLayer();
    cairo_surface_flush(g.surf);
    // the scrim's frozen picture this frame did not draw (the dialog closed) is let go
    if (g.frozen && g.frozenUsed != g.frame) { cairo_surface_destroy(g.frozen); g.frozen = nullptr; g.frozenKey = 0; }
}

bool Gfx::Pixels(std::vector<uint8_t>& bgra, UINT& w, UINT& h) {
    w = h = 0;
    if (!im_ || !im_->surf) return false;
    GfxCairo& g = *im_;
    while (!g.layers.empty()) g.PopLayer();
    cairo_surface_flush(g.surf);
    w = (UINT)g.pw; h = (UINT)g.ph;
    bgra.resize((size_t)w * h * 4);
    const int stride = cairo_image_surface_get_stride(g.surf);
    const uint8_t* src = cairo_image_surface_get_data(g.surf);
    for (UINT y = 0; y < h; ++y) memcpy(&bgra[(size_t)y * w * 4], src + (size_t)y * stride, (size_t)w * 4);
    return true;
}

bool Gfx::SavePng(const wstring& path) {
    if (!im_ || !im_->surf) return false;
    while (!im_->layers.empty()) im_->PopLayer();
    cairo_surface_flush(im_->surf);
    return cairo_surface_write_to_png(im_->surf, U8(path).c_str()) == CAIRO_STATUS_SUCCESS;
}

// The caches let go of what has not been used for a while, and the surfaces of images the art cache let go
void Gfx::Trim() {
    GfxCairo& g = *im_;
    for (auto it = g.bitmaps.begin(); it != g.bitmaps.end();)
        if (it->second.img.expired()) { GfxCairo::FreeBmp(it->second); it = g.bitmaps.erase(it); } else ++it;
    for (auto it = g.shadows.begin(); it != g.shadows.end();)
        if (it->second.used + 240 < g.frame) { cairo_surface_destroy(it->second.s); it = g.shadows.erase(it); } else ++it;
    if (g.layouts.size() > 3000) {
        for (auto it = g.layouts.begin(); it != g.layouts.end();)
            if (it->second.used + 240 < g.frame) { GfxCairo::FreeLayout(it->second); it = g.layouts.erase(it); } else ++it;
        if (g.layouts.size() > 6000) { for (auto& l : g.layouts) GfxCairo::FreeLayout(l.second); g.layouts.clear(); }
    }
}

// ---- shapes ----------------------------------------------------------------------------------------------------------------
#define LIVE (!dry && im_ && im_->cr)

void Gfx::SetOffset(float dx, float dy) {
    ox_ = dx; oy_ = dy;
    if (im_ && im_->cr) ApplyMatrix(*im_, scale, dx, dy);
}
void Gfx::Clear(Col c) {
    if (!LIVE) return;
    cairo_save(im_->cr);
    cairo_set_operator(im_->cr, CAIRO_OPERATOR_SOURCE);
    SetSrc(im_->cr, c);
    cairo_paint(im_->cr);
    cairo_restore(im_->cr);
}
void Gfx::Fill(const Rc& r, Col c) {
    if (!LIVE) return;
    cairo_t* cr = im_->cr;
    cairo_new_path(cr);
    cairo_rectangle(cr, r.x, r.y, r.w, r.h);
    SetSrc(cr, c);
    cairo_fill(cr);
}
void Gfx::FillRound(const Rc& r, float rad, Col c) {
    if (!LIVE) return;
    if (rad <= 0) { Fill(r, c); return; }
    RoundPath(im_->cr, r, rad, rad, rad, rad);
    SetSrc(im_->cr, c);
    cairo_fill(im_->cr);
}
void Gfx::FillRound4(const Rc& r, float tl, float tr, float br, float bl, Col c) {
    if (!LIVE) return;
    RoundPath(im_->cr, r, tl, tr, br, bl);
    SetSrc(im_->cr, c);
    cairo_fill(im_->cr);
}
void Gfx::StrokeRound(const Rc& r, float rad, Col c, float width) {
    if (!LIVE) return;
    const Rc s = r.in(width / 2);  // the stroke lies inside r
    const float q = std::max(0.f, rad - width / 2);
    RoundPath(im_->cr, s, q, q, q, q);
    SetSrc(im_->cr, c);
    cairo_set_line_width(im_->cr, width);
    cairo_stroke(im_->cr);
}

void Gfx::GradStops(const Rc& r, float rad, const std::vector<std::pair<float, Col>>& stops, float x0, float y0, float x1, float y1, float opacity) {
    if (!LIVE || stops.empty()) return;
    cairo_t* cr = im_->cr;
    const float op = std::max(0.f, std::min(1.f, opacity));
    cairo_pattern_t* p = cairo_pattern_create_linear(r.x + x0 * r.w, r.y + y0 * r.h, r.x + x1 * r.w, r.y + y1 * r.h);
    for (auto& s : stops) cairo_pattern_add_color_stop_rgba(p, s.first, s.second.r, s.second.g, s.second.b, s.second.a * op);  // opacity scales every stop
    if (rad > 0) RoundPath(cr, r, rad, rad, rad, rad);
    else { cairo_new_path(cr); cairo_rectangle(cr, r.x, r.y, r.w, r.h); }
    cairo_set_source(cr, p);
    cairo_fill(cr);
    cairo_pattern_destroy(p);
}
void Gfx::Grad(const Rc& r, float rad, Col a, Col b, float x0, float y0, float x1, float y1) { GradStops(r, rad, {{0.f, a}, {1.f, b}}, x0, y0, x1, y1); }

void Gfx::Line(float x1, float y1, float x2, float y2, Col c, float width) {
    if (!LIVE) return;
    cairo_t* cr = im_->cr;
    cairo_new_path(cr);
    cairo_move_to(cr, x1, y1);
    cairo_line_to(cr, x2, y2);
    SetSrc(cr, c);
    cairo_set_line_width(cr, width);
    cairo_set_line_cap(cr, CAIRO_LINE_CAP_BUTT);
    cairo_stroke(cr);
}
void Gfx::Circle(float cx, float cy, float rad, Col c) {
    if (!LIVE) return;
    cairo_new_path(im_->cr);
    cairo_arc(im_->cr, cx, cy, rad, 0, 2 * kPi);
    SetSrc(im_->cr, c);
    cairo_fill(im_->cr);
}
void Gfx::Ring(float cx, float cy, float rad, Col c, float width) {
    if (!LIVE) return;
    cairo_new_path(im_->cr);
    cairo_arc(im_->cr, cx, cy, rad, 0, 2 * kPi);
    SetSrc(im_->cr, c);
    cairo_set_line_width(im_->cr, width);
    cairo_stroke(im_->cr);
}

// A soft shadow as CSS draws one: the rounded shape blurred by a Gaussian of `blur` / 2 standard deviation, at the strength
// the Windows version's stack of fading layers reaches deep inside (0.73 of `alpha`). Layers of that few alpha do not add up
// in 8-bit pixels here (each rounds to nothing), so the shadow is made once as a blurred mask (three box passes) and kept
// by its shape and strength (GfxCairo::shadows); a draw is one paint. `covered` has nothing to save: one paint either way.
void Gfx::Shadow(const Rc& r, float rad, float blur, float alpha, float dy, bool) {
    if (!LIVE || alpha <= 0 || blur <= 0 || r.w <= 0 || r.h <= 0) return;
    GfxCairo& g = *im_;
    const float peak = std::min(0.95f, alpha * 0.73f), sd = blur / 2 * scale;
    const int m = (int)std::ceil(sd * 3), tw = (int)std::ceil(r.w * scale), th = (int)std::ceil(r.h * scale);
    const std::array<int, 5> key{tw, th, (int)std::lround(rad * scale * 4), (int)std::lround(sd * 10), (int)std::lround(peak * 1000)};
    auto it = g.shadows.find(key);
    if (it == g.shadows.end()) {
        if (g.shadows.size() >= 64) { for (auto& e : g.shadows) cairo_surface_destroy(e.second.s); g.shadows.clear(); }
        cairo_surface_t* t = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, tw + 2 * m, th + 2 * m);
        cairo_t* cr = cairo_create(t);
        const float rr = rad * scale;
        RoundPath(cr, Rc{(float)m, (float)m, r.w * scale, r.h * scale}, rr, rr, rr, rr);
        cairo_set_source_rgba(cr, 0, 0, 0, 1);
        cairo_fill(cr);
        cairo_destroy(cr);
        cairo_surface_flush(t);
        const int w = tw + 2 * m, h = th + 2 * m, stride = cairo_image_surface_get_stride(t);
        uint8_t* px = cairo_image_surface_get_data(t);
        BoxBlur(px, w, h, stride, sd);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) { uint8_t* q = px + (size_t)y * stride + (size_t)x * 4; q[3] = (uint8_t)std::lround(q[3] * peak); }  // black: only alpha
        cairo_surface_mark_dirty(t);
        it = g.shadows.emplace(key, GfxCairo::Shadowed{t, g.frame}).first;
    }
    it->second.used = g.frame;
    cairo_save(g.cr);
    cairo_translate(g.cr, r.x, r.y + dy);
    cairo_scale(g.cr, 1 / scale, 1 / scale);
    cairo_set_source_surface(g.cr, it->second.s, -m, -m);
    cairo_paint(g.cr);
    cairo_restore(g.cr);
}

void Gfx::Glow(float cx, float cy, float rx, float ry, Col c) {
    if (!LIVE || rx <= 0 || ry <= 0) return;
    cairo_t* cr = im_->cr;
    cairo_save(cr);
    cairo_translate(cr, cx, cy);
    cairo_scale(cr, 1, ry / rx);  // an ellipse is a circle in the squeezed space
    cairo_pattern_t* p = cairo_pattern_create_radial(0, 0, 0, 0, 0, rx);
    cairo_pattern_add_color_stop_rgba(p, 0, c.r, c.g, c.b, c.a);
    cairo_pattern_add_color_stop_rgba(p, 0.45, c.r, c.g, c.b, c.a * 0.45f);
    cairo_pattern_add_color_stop_rgba(p, 1, c.r, c.g, c.b, 0);
    cairo_new_path(cr);
    cairo_arc(cr, 0, 0, rx, 0, 2 * kPi);
    cairo_set_source(cr, p);
    cairo_fill(cr);
    cairo_pattern_destroy(p);
    cairo_restore(cr);
}

// ---- clips and layers ------------------------------------------------------------------------------------------------------
// A clip is a cairo_save; the matrix is set again when it is popped (a SetOffset made inside it must outlive it, as on Windows).
void Gfx::PushClip(const Rc& r) {
    if (!LIVE) return;
    cairo_save(im_->cr);
    cairo_new_path(im_->cr);
    cairo_rectangle(im_->cr, r.x, r.y, r.w, r.h);
    cairo_clip(im_->cr);
    im_->layers.push_back({false, 1});
}
void Gfx::PopClip() {
    if (!LIVE) return;
    im_->PopLayer();
    ApplyMatrix(*im_, scale, ox_, oy_);
}
// rad < 0: only the top corners are rounded
void Gfx::PushRoundClip(const Rc& r, float rad) {
    if (!LIVE) return;
    const float a = std::fabs(rad);
    cairo_save(im_->cr);
    RoundPath(im_->cr, r, a, a, rad < 0 ? 0 : a, rad < 0 ? 0 : a);
    cairo_clip(im_->cr);
    im_->layers.push_back({false, 1});
}
void Gfx::PopRoundClip() {
    if (!LIVE) return;
    im_->PopLayer();
    ApplyMatrix(*im_, scale, ox_, oy_);
}
// Everything drawn until PopFade goes to a group that is painted back at `opacity`
void Gfx::PushFade(float opacity) {
    if (!LIVE) return;
    cairo_push_group(im_->cr);
    im_->layers.push_back({true, std::min(1.f, std::max(0.f, opacity))});
}
void Gfx::PopFade() { PopRoundClip(); }

// ---- text ----------------------------------------------------------------------------------------------------------------------
namespace {
using CachedLayout = GfxCairo::CachedLayout;

// Wrapped text whose last line is a single word is wrapped again at the widest width that brings a second word down to it
// without adding a line (the Windows version's Balance, line for line); returns how far the narrower layout moves.
float Balance(PangoLayout* l, const std::string& u, float w, float boxH, const TS& s) {
    if (!s.wrap || u.find('\n') != std::string::npos) return 0;
    const int n = pango_layout_get_line_count(l);
    if (n < 2 || n > 6) return 0;
    PangoRectangle lr;
    pango_layout_get_extents(l, nullptr, &lr);
    if (pango_layout_is_ellipsized(l) || lr.height / (float)PANGO_SCALE > boxH + 0.5f) return 0;  // cut off: left as it is
    auto good = [&]() {
        if (pango_layout_get_line_count(l) != n) return false;
        for (int i = 0; i + 1 < n; ++i) {
            PangoLayoutLine* ln = pango_layout_get_line_readonly(l, i);
            const int end = ln->start_index + ln->length;
            if (end > 0 && u[(size_t)end - 1] == '-') return false;  // no word broken at its hyphen
        }
        PangoLayoutLine* last = pango_layout_get_line_readonly(l, n - 1);
        std::string t = u.substr((size_t)last->start_index, (size_t)last->length);
        while (!t.empty() && t.back() == ' ') t.pop_back();
        // two words: a space, or NoWidow's no-break space
        return t.find(' ') != std::string::npos || t.find("\xC2\xA0") != std::string::npos;
    };
    auto setW = [&](float v) { pango_layout_set_width(l, (int)(v * PANGO_SCALE)); };
    if (good()) return 0;
    float lo = w * (1 - 0.5f / n), hi = w;
    setW(lo);
    if (!good()) { setW(w); return 0; }
    for (int i = 0; i < 8; ++i) {  // the widest width that is still good
        const float mid = (lo + hi) / 2;
        setW(mid);
        if (good()) lo = mid; else hi = mid;
    }
    setW(lo);
    const float slack = w - lo;
    return s.align == 1 ? slack / 2 : s.align == 2 ? slack : 0;
}
}  // namespace

// leadLen < 0: not a lead layout. A layout for its text, style (not its colour: that is set as it is drawn), box and lead.
static CachedLayout MakeLayout(GfxCairo& g, const wstring& t, float w, float h, const TS& s, bool balance, long leadLen, int leadWeight) {
    CachedLayout c;
    c.u8 = U8(t);
    PangoLayout* l = pango_layout_new(g.ctx);
    PangoFontDescription* d = pango_font_description_new();
    pango_font_description_set_family(d, g.fam[s.face].c_str());
    pango_font_description_set_weight(d, (PangoWeight)s.weight);
    pango_font_description_set_absolute_size(d, s.size * PANGO_SCALE);
    pango_layout_set_font_description(l, d);
    pango_font_description_free(d);
    pango_layout_set_text(l, c.u8.data(), (int)c.u8.size());
    pango_layout_set_width(l, (int)(std::min(std::max(1.f, w), 1e6f) * PANGO_SCALE));
    pango_layout_set_alignment(l, s.align == 1 ? PANGO_ALIGN_CENTER : s.align == 2 ? PANGO_ALIGN_RIGHT : PANGO_ALIGN_LEFT);
    pango_layout_set_wrap(l, PANGO_WRAP_WORD_CHAR);
    if (!s.wrap) {
        pango_layout_set_ellipsize(l, PANGO_ELLIPSIZE_END);  // height stays -1: one line a paragraph
    } else if (s.trim) {
        pango_layout_set_ellipsize(l, PANGO_ELLIPSIZE_END);
        pango_layout_set_height(l, (int)(std::min(std::max(1.f, h), 1e6f) * PANGO_SCALE));  // the lines that fit the box
    }
    c.base = pango_attr_list_new();
    if (s.spacing != 0) pango_attr_list_insert(c.base, pango_attr_letter_spacing_new((int)std::lround(s.spacing * PANGO_SCALE)));
    if (s.lineH > 0) pango_attr_list_insert(c.base, pango_attr_line_height_new_absolute((int)std::lround(s.lineH * PANGO_SCALE)));
    if (s.underline) pango_attr_list_insert(c.base, pango_attr_underline_new(PANGO_UNDERLINE_SINGLE));
    if (leadLen >= 0) {
        c.leadBytes = (int)U8(t.substr(0, (size_t)leadLen)).size();
        PangoAttribute* wa = pango_attr_weight_new((PangoWeight)leadWeight);
        wa->start_index = 0; wa->end_index = (guint)c.leadBytes;
        pango_attr_list_insert(c.base, wa);
    }
    pango_layout_set_attributes(l, c.base);
    c.l = l;
    if (balance) c.dx = Balance(l, c.u8, w, h, s);
    PangoRectangle lr;
    pango_layout_get_extents(l, nullptr, &lr);
    c.w = lr.width / (float)PANGO_SCALE;
    c.h = lr.height / (float)PANGO_SCALE;
    c.used = g.frame;
    return c;
}

using LayoutKey = std::tuple<wstring, int, int, int, int, int, int>;
static int Q64(float f) { return (int)std::lround(std::min(f, 1e6f) * 64); }
static int StyleFlags(const TS& s, bool balance) {
    return (int)s.face | s.align << 3 | s.valign << 5 | (s.wrap ? 1 << 7 : 0) | (s.trim ? 1 << 8 : 0) | (s.underline ? 1 << 9 : 0) | (balance ? 1 << 10 : 0) | s.weight << 12;
}

// Kept in layouts (Trim lets go of those not used for a while); the key is the Windows version's
static const CachedLayout* Cached(GfxCairo& g, const wstring& t, float w, float h, const TS& s, bool balance) {
    const LayoutKey key{t, Q64(w), Q64(h), (int)std::lround(s.size * 100), StyleFlags(s, balance), (int)std::lround(s.spacing * 100), (int)std::lround(s.lineH * 100)};
    auto it = g.layouts.find(key);
    if (it != g.layouts.end()) { it->second.used = g.frame; return &it->second; }
    CachedLayout c = MakeLayout(g, t, w, h, s, balance, -1, 0);
    return &(g.layouts[key] = c);
}

// Display text (the Display face, text of 20 DIPs or more) is grayscale on Windows's screen; here every frame is grayscale
// already (Init), so Text and TextLead draw the same everywhere.
static void ShowLayout(GfxCairo& g, PangoLayout* l, float x, float y, Col c) {
    cairo_move_to(g.cr, x, y);
    SetSrc(g.cr, c);
    pango_cairo_update_layout(g.cr, l);
    pango_cairo_show_layout(g.cr, l);
}
static float VAlign(const CachedLayout& c, const Rc& r, const TS& s) { return s.valign == 1 ? (r.h - c.h) / 2 : s.valign == 2 ? r.h - c.h : 0; }

float Gfx::Text(const wstring& t, const Rc& r, const TS& s) {
    if (t.empty() || !im_) return 0;
    const CachedLayout* c = Cached(*im_, t, r.w, std::max(r.h, 1.f), s, true);
    if (!dry && im_->cr) ShowLayout(*im_, c->l, r.x + c->dx, r.y + VAlign(*c, r, s), s.color);
    return c->h;
}
float Gfx::TextW(const wstring& t, const TS& s) {
    if (t.empty() || !im_) return 0;
    TS n = s; n.wrap = false; n.align = 0; n.valign = 0; n.trim = false; n.underline = false; n.lineH = 0;
    return Cached(*im_, t, 100000, 1000, n, false)->w;
}
float Gfx::TextH(const wstring& t, float width, const TS& s) {
    if (t.empty() || !im_) return 0;
    TS n = s; n.valign = 0; n.underline = false;
    return Cached(*im_, t, width, 100000, n, false)->h;
}
void Gfx::TextEnd(const wstring& t, float width, const TS& s, float& x, float& y) {
    x = y = 0;
    if (t.empty() || !im_) return;
    TS n = s; n.valign = 0; n.align = 0;
    const CachedLayout* c = Cached(*im_, t, width, 100000, n, true);  // balanced: where Text draws it
    size_t i = c->u8.size() - 1;
    while (i > 0 && (((unsigned char)c->u8[i]) & 0xC0) == 0x80) --i;  // the last character's first byte
    PangoRectangle pos;
    pango_layout_index_to_pos(c->l, (int)i, &pos);
    x = (pos.x + pos.width) / (float)PANGO_SCALE;
    y = pos.y / (float)PANGO_SCALE;
}

// Warn once for a glyph the table has nothing for (a new icon constant that was not mapped)
void Gfx::Icon(wchar_t glyph, float cx, float cy, float size, Col c) {
    wchar_t mapped = 0;
    for (const IconPair& p : kIcons) if (p.segoe == glyph) { mapped = p.fluent; break; }
    if (!mapped) {
        static std::set<wchar_t> warned;
        if (warned.insert(glyph).second) fprintf(stderr, "wronggpu: no Fluent System Icons glyph mapped for U+%04X; drawn as nothing\n", (unsigned)glyph);
        return;
    }
    TS s; s.face = F_ICON; s.size = size; s.color = c; s.align = 1; s.valign = 1;
    Text(wstring(1, mapped), Rc{cx - size, cy - size, size * 2, size * 2}, s);
}

// The lead's colour is an attribute set as the layout is drawn (a layout outlives its colour, as on Windows: the cache key
// has the lead's length and weight, not its colour).
float Gfx::TextLead(const wstring& t, size_t leadLen, Col leadC, int leadWeight, const Rc& r, const TS& s) {
    if (t.empty() || !im_) return 0;
    GfxCairo& g = *im_;
    const size_t lead = std::min(leadLen, t.size());
    const wstring k = t + L'\x1' + std::to_wstring(lead) + L',' + std::to_wstring(leadWeight);
    const LayoutKey key{k, Q64(r.w), Q64(std::max(r.h, 1.f)), (int)std::lround(s.size * 100), StyleFlags(s, true) | 1 << 11, (int)std::lround(s.spacing * 100), (int)std::lround(s.lineH * 100)};
    auto it = g.layouts.find(key);
    if (it == g.layouts.end()) it = g.layouts.emplace(key, MakeLayout(g, t, r.w, std::max(r.h, 1.f), s, true, (long)lead, leadWeight)).first;
    CachedLayout& c = it->second;
    c.used = g.frame;
    if (!dry && g.cr) {
        if (c.leadCol.r != leadC.r || c.leadCol.g != leadC.g || c.leadCol.b != leadC.b || c.leadCol.a != leadC.a) {
            PangoAttrList* al = pango_attr_list_copy(c.base);
            auto u16 = [](float f) { return (guint16)std::lround(std::max(0.f, std::min(1.f, f)) * 65535); };
            PangoAttribute* fg = pango_attr_foreground_new(u16(leadC.r), u16(leadC.g), u16(leadC.b));
            PangoAttribute* fa = pango_attr_foreground_alpha_new(u16(leadC.a));
            fg->start_index = fa->start_index = 0;
            fg->end_index = fa->end_index = (guint)c.leadBytes;
            pango_attr_list_insert(al, fg);
            pango_attr_list_insert(al, fa);
            pango_layout_set_attributes(c.l, al);
            pango_attr_list_unref(al);
            c.leadCol = leadC;
        }
        ShowLayout(g, c.l, r.x + c.dx, r.y + VAlign(c, r, s), s.color);
    }
    return c.h;
}

// ---- images --------------------------------------------------------------------------------------------------------------------
// One surface per image, made once (premultiplied BGRA is cairo's ARGB32 as it is); those not drawn this frame go when many are kept
static GfxCairo::Bmp* Bitmap(GfxCairo& g, const ImageP& img) {
    if (!img || img->w <= 0 || img->h <= 0 || img->px.size() < (size_t)img->w * img->h * 4) return nullptr;
    auto it = g.bitmaps.find(img.get());
    if (it != g.bitmaps.end()) {
        const std::weak_ptr<const Image>& w = it->second.img;
        if (!w.owner_before(img) && !img.owner_before(w)) { it->second.used = g.frame; return &it->second; }
        GfxCairo::FreeBmp(it->second);  // another image at a gone one's address
        g.bitmaps.erase(it);
    }
    if (g.bitmaps.size() >= 128)
        for (auto k = g.bitmaps.begin(); k != g.bitmaps.end();) {
            if (k->second.used != g.frame) { GfxCairo::FreeBmp(k->second); k = g.bitmaps.erase(k); } else ++k;
        }
    GfxCairo::Bmp b;
    b.img = img;
    b.s = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, img->w, img->h);
    if (cairo_surface_status(b.s) != CAIRO_STATUS_SUCCESS) { cairo_surface_destroy(b.s); return nullptr; }
    cairo_surface_flush(b.s);
    const int stride = cairo_image_surface_get_stride(b.s);
    uint8_t* dst = cairo_image_surface_get_data(b.s);
    for (int y = 0; y < img->h; ++y) memcpy(dst + (size_t)y * stride, &img->px[(size_t)y * img->w * 4], (size_t)img->w * 4);
    cairo_surface_mark_dirty(b.s);
    b.used = g.frame;
    return &(g.bitmaps[img.get()] = b);
}

// src (sx, sy, sw, sh) of the surface stretched over dst; extend PAD so an edge pixel does not blend with nothing
static void DrawSurf(cairo_t* cr, cairo_surface_t* s, const Rc& dst, float sx, float sy, float sw, float sh, float opacity, cairo_filter_t filter) {
    if (dst.w <= 0 || dst.h <= 0 || sw <= 0 || sh <= 0) return;
    cairo_save(cr);
    cairo_new_path(cr);
    cairo_rectangle(cr, dst.x, dst.y, dst.w, dst.h);
    cairo_clip(cr);
    cairo_translate(cr, dst.x, dst.y);
    cairo_scale(cr, dst.w / sw, dst.h / sh);
    cairo_translate(cr, -sx, -sy);
    cairo_set_source_surface(cr, s, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), filter);
    cairo_pattern_set_extend(cairo_get_source(cr), CAIRO_EXTEND_PAD);
    cairo_paint_with_alpha(cr, std::max(0.f, std::min(1.f, opacity)));
    cairo_restore(cr);
}

void Gfx::ImageCover(const ImageP& img, const Rc& r, float rad, float focusY, float opacity, bool topOnly) {
    if (!LIVE) return;
    GfxCairo::Bmp* b = Bitmap(*im_, img);
    if (!b) return;
    const float k = std::max(r.w / img->w, r.h / img->h);
    const float sw = r.w / k, sh = r.h / k;
    const float sx = (img->w - sw) / 2, sy = (img->h - sh) * focusY;
    if (rad > 0) PushRoundClip(r, topOnly ? -rad : rad);
    DrawSurf(im_->cr, b->s, r, sx, sy, sw, sh, opacity, CAIRO_FILTER_BEST);
    if (rad > 0) PopRoundClip();
}

// The alpha of the image at device size, blurred, in the art-shade colour: ImageContain's drop shadow (CSS drop-shadow). Made
// once per bitmap and size and kept (Bmp::shadow); `m` pixels of margin round it for the blur.
static cairo_surface_t* MakeShadow(GfxCairo::Bmp& b, int tw, int th, int m, float sigma, Col c) {
    cairo_surface_t* t = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, tw + 2 * m, th + 2 * m);
    cairo_t* cr = cairo_create(t);
    cairo_translate(cr, m, m);
    cairo_scale(cr, (double)tw / cairo_image_surface_get_width(b.s), (double)th / cairo_image_surface_get_height(b.s));
    cairo_set_source_surface(cr, b.s, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_BEST);
    cairo_paint(cr);
    cairo_destroy(cr);
    cairo_surface_flush(t);
    const int w = cairo_image_surface_get_width(t), h = cairo_image_surface_get_height(t), stride = cairo_image_surface_get_stride(t);
    uint8_t* p = cairo_image_surface_get_data(t);
    BoxBlur(p, w, h, stride, sigma);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t* q = p + (size_t)y * stride + (size_t)x * 4;
            const float a = q[3] / 255.f * 0.75f;  // premultiplied: the colour scales with the alpha
            q[0] = (uint8_t)std::lround(c.b * a * 255); q[1] = (uint8_t)std::lround(c.g * a * 255); q[2] = (uint8_t)std::lround(c.r * a * 255); q[3] = (uint8_t)std::lround(a * 255);
        }
    cairo_surface_mark_dirty(t);
    return t;
}

void Gfx::ImageContain(const ImageP& img, const Rc& box, int ax, int ay, float opacity, float shadow) {
    if (!LIVE) return;
    GfxCairo& g = *im_;
    GfxCairo::Bmp* b = Bitmap(g, img);
    if (!b) return;
    const float k = std::min(box.w / img->w, box.h / img->h);
    const float w = img->w * k, h = img->h * k;
    const float x = box.x + (box.w - w) * ax / 2, y = box.y + (box.h - h) * ay / 2;
    if (shadow > 0 && k > 0) {
        const int tw = std::max(1, (int)std::lround(w * scale)), th = std::max(1, (int)std::lround(h * scale));
        const float sd = shadow / 2 * scale;  // pixels
        const int m = (int)std::ceil(sd * 3), sdq = (int)std::lround(sd * 10);
        if (!b->shadow || b->shadowKey != std::make_tuple(tw, th, sdq)) {
            if (b->shadow) cairo_surface_destroy(b->shadow);
            b->shadow = MakeShadow(*b, tw, th, m, sd, pal.artShade);
            b->shadowKey = std::make_tuple(tw, th, sdq);
        }
        cairo_save(g.cr);
        cairo_translate(g.cr, x - m / scale, y + shadow / 4 - m / scale);
        cairo_scale(g.cr, 1 / scale, 1 / scale);
        cairo_set_source_surface(g.cr, b->shadow, 0, 0);
        cairo_paint_with_alpha(g.cr, std::max(0.f, std::min(1.f, opacity)));
        cairo_restore(g.cr);
    }
    DrawSurf(g.cr, b->s, Rc{x, y, w, h}, 0, 0, (float)img->w, (float)img->h, opacity, CAIRO_FILTER_BEST);
}

void Gfx::ImageNearest(const ImageP& img, const Rc& r, float rad, float opacity) {
    if (!LIVE) return;
    GfxCairo::Bmp* b = Bitmap(*im_, img);
    if (!b) return;
    if (rad > 0) PushRoundClip(r, rad);
    DrawSurf(im_->cr, b->s, r, 0, 0, (float)img->w, (float)img->h, opacity, CAIRO_FILTER_NEAREST);
    if (rad > 0) PopRoundClip();
}

// ---- backdrop and the frozen frame ------------------------------------------------------------------------------------------------
// The target's pixels under r (plus three deviations round it for the blur to sample, the Windows version's margin) copied and
// blurred on the CPU (three box passes, the edge pixel repeated), then drawn back through r's rounded shape at `opacity`.
bool Gfx::Backdrop(const Rc& r, float rad, float blur, float opacity) {
    if (dry) return true;
    if (!im_ || !im_->cr || !im_->layers.empty() || blur <= 0.05f || opacity <= 0) return false;
    GfxCairo& g = *im_;
    const float m = blur * 3, wx = r.x + ox_, wy = r.y + oy_;  // window DIPs
    const int x0 = (int)std::max(0.f, std::floor((wx - m) * scale)), y0 = (int)std::max(0.f, std::floor((wy - m) * scale));
    const int x1 = (int)std::min((float)g.pw, std::ceil((wx + r.w + m) * scale)), y1 = (int)std::min((float)g.ph, std::ceil((wy + r.h + m) * scale));
    if (x1 <= x0 + 1 || y1 <= y0 + 1) return false;
    cairo_surface_flush(g.surf);
    cairo_surface_t* t = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, x1 - x0, y1 - y0);
    if (cairo_surface_status(t) != CAIRO_STATUS_SUCCESS) { cairo_surface_destroy(t); return false; }
    const int ss = cairo_image_surface_get_stride(g.surf), ts = cairo_image_surface_get_stride(t);
    const uint8_t* src = cairo_image_surface_get_data(g.surf);
    uint8_t* dst = cairo_image_surface_get_data(t);
    for (int y = y0; y < y1; ++y) memcpy(dst + (size_t)(y - y0) * ts, src + (size_t)y * ss + (size_t)x0 * 4, (size_t)(x1 - x0) * 4);
    BoxBlur(dst, x1 - x0, y1 - y0, ts, blur * scale);
    cairo_surface_mark_dirty(t);
    cairo_save(g.cr);
    if (rad > 0) RoundPath(g.cr, r, rad, rad, rad, rad);
    else { cairo_new_path(g.cr); cairo_rectangle(g.cr, r.x, r.y, r.w, r.h); }
    cairo_clip(g.cr);
    cairo_identity_matrix(g.cr);  // the copy is in device pixels
    cairo_set_source_surface(g.cr, t, x0, y0);
    cairo_paint_with_alpha(g.cr, std::min(1.f, opacity));
    cairo_restore(g.cr);
    cairo_surface_destroy(t);
    return true;
}

bool Gfx::DrawFrozen(unsigned long long key) {
    if (dry) return true;
    if (!im_ || !im_->cr || !im_->frozen || im_->frozenKey != key || !im_->layers.empty()) return false;
    GfxCairo& g = *im_;
    if (cairo_image_surface_get_width(g.frozen) != g.pw || cairo_image_surface_get_height(g.frozen) != g.ph) return false;
    g.frozenUsed = g.frame;
    cairo_save(g.cr);
    cairo_identity_matrix(g.cr);
    cairo_set_source_surface(g.cr, g.frozen, 0, 0);
    cairo_pattern_set_filter(cairo_get_source(g.cr), CAIRO_FILTER_NEAREST);
    cairo_paint(g.cr);
    cairo_restore(g.cr);
    return true;
}
void Gfx::Freeze(unsigned long long key) {
    if (!LIVE || !im_->layers.empty()) return;
    GfxCairo& g = *im_;
    cairo_surface_flush(g.surf);
    if (g.frozen && (cairo_image_surface_get_width(g.frozen) != g.pw || cairo_image_surface_get_height(g.frozen) != g.ph)) { cairo_surface_destroy(g.frozen); g.frozen = nullptr; }
    if (!g.frozen) g.frozen = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, g.pw, g.ph);
    cairo_t* cr = cairo_create(g.frozen);
    cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
    cairo_set_source_surface(cr, g.surf, 0, 0);
    cairo_paint(cr);
    cairo_destroy(cr);
    g.frozenKey = key;
    g.frozenUsed = g.frame;
}

}  // namespace wg::app::lui
