// SPDX-License-Identifier: MIT
// WrongGPU launcher window: drawing with Direct2D 1.1, DirectWrite and WIC (ui_shell.h). One class for the window's
// swap chain (a Direct3D 11 flip-model swap chain drawn through a Direct2D device context) and for offscreen renders
// into a WIC bitmap (ui-dump, ui-render), so both paint the same frame.
// Written for this project.
#include "app/ui_shell.h"

#include <d2d1effects.h>

#ifdef _MSC_VER
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#endif

namespace wg::app::lui {

template <class T> static void SafeRelease(T*& p) { if (p) { p->Release(); p = nullptr; } }
static D2D1_COLOR_F DC(Col c) { return D2D1::ColorF(c.r, c.g, c.b, c.a); }
static D2D1_RECT_F DR(const Rc& r) { return D2D1::RectF(r.x, r.y, r.r(), r.b()); }

Gfx::~Gfx() {
    Release();
    for (auto& l : layouts_) SafeRelease(l.second.l);
    layouts_.clear();
    for (auto& p : paths_) SafeRelease(p.second.p);
    paths_.clear();
    for (auto& sg : signs_) SafeRelease(sg.second);
    signs_.clear();
    for (auto& f : formats_) f.second->Release();
    formats_.clear();
    SafeRelease(d2d_); SafeRelease(dw); SafeRelease(wic);
}

bool Gfx::Init() {
    if (d2d_) return true;
    D2D1_FACTORY_OPTIONS o{};
    if (FAILED(D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, __uuidof(ID2D1Factory1), &o, (void**)&d2d_))) return false;
    if (FAILED(DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), (IUnknown**)&dw))) return false;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic)))) return false;
    // Segoe UI Variable's optical sizes (Windows 11): its Text family for body text and controls, its Display family for
    // titles and headings (the named instances of the font's opsz axis), else Segoe UI; Cascadia Mono, else Consolas;
    // Segoe Fluent Icons, else Segoe MDL2 Assets.
    IDWriteFontCollection* sys = nullptr;
    dw->GetSystemFontCollection(&sys);
    auto has = [&](const wchar_t* f) { UINT32 i = 0; BOOL e = FALSE; return sys && SUCCEEDED(sys->FindFamilyName(f, &i, &e)) && e; };
    families_[F_TEXT] = has(L"Segoe UI Variable Text") ? L"Segoe UI Variable Text" : L"Segoe UI";
    families_[F_DISPLAY] = has(L"Segoe UI Variable Display") ? L"Segoe UI Variable Display" : L"Segoe UI";
    families_[F_MONO] = has(L"Cascadia Mono") ? L"Cascadia Mono" : L"Consolas";
    families_[F_ICON] = has(L"Segoe Fluent Icons") ? L"Segoe Fluent Icons" : L"Segoe MDL2 Assets";
    iconFont = has(families_[F_ICON].c_str());
    SafeRelease(sys);
    return true;
}

void Gfx::Release() {
    for (auto& b : bitmaps_) { SafeRelease(b.second.b); SafeRelease(b.second.shadow); }
    bitmaps_.clear();
    for (auto& b : grads_) SafeRelease(b.second);
    grads_.clear();
    for (auto& b : glows_) SafeRelease(b.second);
    glows_.clear();
    for (auto*& b : backdropBmp_) SafeRelease(b);
    SafeRelease(blur_); SafeRelease(frozen_); frozenKey_ = 0;
    // the lead layouts outlive the target, but not its brush as their drawing effect: it holds the old device (a lost
    // device's Direct3D device and swap chain would stay alive for the session in layouts not drawn again)
    for (auto& l : layouts_) if (l.second.lead && l.second.l) l.second.l->SetDrawingEffect(nullptr, DWRITE_TEXT_RANGE{0, 0xFFFFFFFFu});
    SafeRelease(leadBrush_);
    SafeRelease(brush_); SafeRelease(dc_);
    if (liveDc_) liveDc_->SetTarget(nullptr);
    SafeRelease(backBmp_); SafeRelease(liveDc_); SafeRelease(d2dDev_); SafeRelease(swap_); SafeRelease(d3d_);
    SafeRelease(wicRt_); SafeRelease(wicBmp_);
    rt = nullptr;
}
void Gfx::DropTarget() { Release(); }

void Gfx::TargetReady() {
    rt->CreateSolidColorBrush(D2D1::ColorF(0, 0, 0, 1), &brush_);
    rt->CreateSolidColorBrush(D2D1::ColorF(0, 0, 0, 1), &leadBrush_);
    rt->QueryInterface(__uuidof(ID2D1DeviceContext), (void**)&dc_);
    rt->SetTextAntialiasMode(offscreen ? D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE : D2D1_TEXT_ANTIALIAS_MODE_DEFAULT);
}

// The live window draws into a flip-model swap chain rather than a Direct2D HWND render target: on some systems the HWND
// target reports success while nothing reaches the screen (the window stays white), and the swap chain is the
// presentation path current Windows composes directly.
bool Gfx::CreateSwapChain(HWND hwnd, UINT w, UINT h, UINT dpi) {
    const UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, nullptr, 0, D3D11_SDK_VERSION, &d3d_, nullptr, nullptr)) &&
        FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags, nullptr, 0, D3D11_SDK_VERSION, &d3d_, nullptr, nullptr)))
        return false;
    IDXGIDevice* dxgi = nullptr; IDXGIAdapter* adapter = nullptr; IDXGIFactory2* factory = nullptr;
    bool ok = SUCCEEDED(d3d_->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi)) &&
              SUCCEEDED(d2d_->CreateDevice(dxgi, &d2dDev_)) &&
              SUCCEEDED(d2dDev_->CreateDeviceContext(D2D1_DEVICE_CONTEXT_OPTIONS_NONE, &liveDc_)) &&
              SUCCEEDED(dxgi->GetAdapter(&adapter)) && SUCCEEDED(adapter->GetParent(__uuidof(IDXGIFactory2), (void**)&factory));
    // at most one frame queued ahead of the display (DXGI's default is three): Present waits once a frame is waiting, so a
    // frame starts from the latest wheel, touchpad and pointer input and reaches the screen on the next refresh, not three later
    if (ok) {
        IDXGIDevice1* d1 = nullptr;
        if (SUCCEEDED(d3d_->QueryInterface(__uuidof(IDXGIDevice1), (void**)&d1))) { d1->SetMaximumFrameLatency(1); d1->Release(); }
    }
    if (ok) {
        DXGI_SWAP_CHAIN_DESC1 d{};
        d.Width = w; d.Height = h;
        d.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        d.SampleDesc.Count = 1;
        d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.BufferCount = 2;
        d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        d.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        // no scaling: while the window is being resized DWM shows the last frame as it was (anchored at the top left) until
        // the next one is presented, instead of stretching it (the content would wobble)
        d.Scaling = DXGI_SCALING_NONE;
        ok = SUCCEEDED(factory->CreateSwapChainForHwnd(d3d_, hwnd, &d, nullptr, nullptr, &swap_));
        if (!ok) { d.Scaling = DXGI_SCALING_STRETCH; ok = SUCCEEDED(factory->CreateSwapChainForHwnd(d3d_, hwnd, &d, nullptr, nullptr, &swap_)); }
        if (ok) factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER | DXGI_MWA_NO_WINDOW_CHANGES);
    }
    SafeRelease(factory); SafeRelease(adapter); SafeRelease(dxgi);
    return ok && BindBackBuffer(dpi);
}

bool Gfx::BindBackBuffer(UINT dpi) {
    IDXGISurface* surface = nullptr;
    if (FAILED(swap_->GetBuffer(0, __uuidof(IDXGISurface), (void**)&surface))) return false;
    const D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(D2D1_BITMAP_OPTIONS_TARGET | D2D1_BITMAP_OPTIONS_CANNOT_DRAW,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_IGNORE), (float)dpi, (float)dpi);
    const bool ok = SUCCEEDED(liveDc_->CreateBitmapFromDxgiSurface(surface, &bp, &backBmp_));
    surface->Release();
    if (ok) liveDc_->SetTarget(backBmp_);
    return ok;
}

bool Gfx::ResizeSwapChain(UINT w, UINT h, UINT dpi) {
    liveDc_->SetTarget(nullptr);
    SafeRelease(backBmp_);
    if (FAILED(swap_->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0))) return false;
    return BindBackBuffer(dpi);
}

// The caches let go of what has not been used for a while (a page left, a game's text scrolled away long ago), and the
// bitmaps of images the art cache let go (SHIP-BAR 1: memory flat; a decoded hero is megabytes in memory and on the GPU)
void Gfx::Trim() {
    for (auto it = bitmaps_.begin(); it != bitmaps_.end();)
        if (it->second.img.expired()) { SafeRelease(it->second.b); SafeRelease(it->second.shadow); it = bitmaps_.erase(it); } else ++it;
    if (layouts_.size() > 3000) {
        for (auto it = layouts_.begin(); it != layouts_.end();)
            if (it->second.used + 240 < frame_) { SafeRelease(it->second.l); it = layouts_.erase(it); } else ++it;
        if (layouts_.size() > 6000) { for (auto& l : layouts_) SafeRelease(l.second.l); layouts_.clear(); }
    }
    if (paths_.size() > 1500) {
        for (auto it = paths_.begin(); it != paths_.end();)
            if (it->second.used + 240 < frame_) { SafeRelease(it->second.p); it = paths_.erase(it); } else ++it;
        if (paths_.size() > 3000) { for (auto& p : paths_) SafeRelease(p.second.p); paths_.clear(); }
    }
}

bool Gfx::BeginHwnd(HWND hwnd, UINT dpi) {
    ++frame_;
    Trim();
    offscreen = false;
    scale = dpi / 96.f;
    RECT rc; GetClientRect(hwnd, &rc);
    const UINT pw = (UINT)std::max<LONG>(1, rc.right), ph = (UINT)std::max<LONG>(1, rc.bottom);
    if (wicRt_) Release();
    if (!liveDc_) {
        if (!CreateSwapChain(hwnd, pw, ph, dpi)) { Release(); return false; }
        rt = liveDc_;
        TargetReady();
    } else {
        DXGI_SWAP_CHAIN_DESC1 d{};
        swap_->GetDesc1(&d);
        if ((d.Width != pw || d.Height != ph) && !ResizeSwapChain(pw, ph, dpi)) { Release(); return false; }
    }
    liveDc_->SetDpi((float)dpi, (float)dpi);
    rt = liveDc_;
    rt->BeginDraw();
    rt->SetTransform(D2D1::Matrix3x2F::Identity());
    ox_ = oy_ = 0;
    return true;
}

void Gfx::Resize(HWND hwnd) {
    if (!swap_) return;
    RECT rc; GetClientRect(hwnd, &rc);
    float dx = 96, dy = 96;
    liveDc_->GetDpi(&dx, &dy);
    if (!ResizeSwapChain((UINT)std::max<LONG>(1, rc.right), (UINT)std::max<LONG>(1, rc.bottom), (UINT)dx)) Release();
}

bool Gfx::BeginOffscreen(int w, int h, float s) {
    ++frame_;
    Trim();
    Release();
    offscreen = true;
    scale = s;
    const UINT pw = (UINT)std::lround(w * s), ph = (UINT)std::lround(h * s);
    if (FAILED(wic->CreateBitmap(pw, ph, GUID_WICPixelFormat32bppPBGRA, WICBitmapCacheOnLoad, &wicBmp_))) return false;
    const D2D1_RENDER_TARGET_PROPERTIES p = D2D1::RenderTargetProperties(D2D1_RENDER_TARGET_TYPE_DEFAULT,
        D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED), 96.f * s, 96.f * s);
    if (FAILED(d2d_->CreateWicBitmapRenderTarget(wicBmp_, p, &wicRt_))) return false;
    rt = wicRt_;
    TargetReady();
    rt->BeginDraw();
    rt->SetTransform(D2D1::Matrix3x2F::Identity());
    ox_ = oy_ = 0;
    return true;
}

void Gfx::EndFrame() {
    if (!rt) return;
    for (size_t i = layers_.size(); i-- > 0;) { rt->PopLayer(); SafeRelease(layers_[i]); }
    layers_.clear();
    const HRESULT hr = rt->EndDraw();
    // the scrim's frozen picture and the backdrop copies this frame did not draw (the dialog closed, the capsule went):
    // let go (at 4K the frozen frame alone is 33 MB)
    if (frozen_ && frozenUsed_ != frame_) { SafeRelease(frozen_); frozenKey_ = 0; }
    for (int i = 0; i < 2; ++i) if (backdropBmp_[i] && backdropUsed_[i] != frame_) SafeRelease(backdropBmp_[i]);
    HRESULT present = S_OK;
    if (!offscreen && swap_ && SUCCEEDED(hr)) present = swap_->Present(1, 0);
    occluded = present == DXGI_STATUS_OCCLUDED;  // nothing on screen: the window paces its frames on a timer meanwhile
    if (hr == D2DERR_RECREATE_TARGET || present == DXGI_ERROR_DEVICE_REMOVED || present == DXGI_ERROR_DEVICE_RESET) Release();
}

bool Gfx::Pixels(std::vector<uint8_t>& bgra, UINT& w, UINT& h) {
    w = h = 0;
    if (!wicBmp_ || FAILED(wicBmp_->GetSize(&w, &h)) || !w || !h) return false;
    bgra.resize((size_t)w * h * 4);
    return SUCCEEDED(wicBmp_->CopyPixels(nullptr, w * 4, (UINT)bgra.size(), bgra.data()));
}

bool Gfx::SavePng(const wstring& path) {
    if (!wicBmp_) return false;
    IWICStream* st = nullptr; IWICBitmapEncoder* enc = nullptr; IWICBitmapFrameEncode* fr = nullptr;
    bool ok = SUCCEEDED(wic->CreateStream(&st)) && SUCCEEDED(st->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
              SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &enc)) && SUCCEEDED(enc->Initialize(st, WICBitmapEncoderNoCache)) &&
              SUCCEEDED(enc->CreateNewFrame(&fr, nullptr)) && SUCCEEDED(fr->Initialize(nullptr));
    if (ok) {
        UINT w = 0, h = 0; wicBmp_->GetSize(&w, &h);
        WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
        ok = SUCCEEDED(fr->SetSize(w, h)) && SUCCEEDED(fr->SetPixelFormat(&fmt));
        // the frame is opaque: premultiplied = straight
        ok = ok && SUCCEEDED(fr->WriteSource(wicBmp_, nullptr)) && SUCCEEDED(fr->Commit()) && SUCCEEDED(enc->Commit());
    }
    SafeRelease(fr); SafeRelease(enc); SafeRelease(st);
    return ok;
}

ID2D1SolidColorBrush* Gfx::B(Col c) { brush_->SetColor(DC(c)); return brush_; }

void Gfx::SetOffset(float dx, float dy) { ox_ = dx; oy_ = dy; rt->SetTransform(D2D1::Matrix3x2F::Translation(dx, dy)); }
void Gfx::Clear(Col c) { if (!dry) rt->Clear(DC(c)); }
void Gfx::Fill(const Rc& r, Col c) { if (!dry) rt->FillRectangle(DR(r), B(c)); }
void Gfx::FillRound(const Rc& r, float rad, Col c) {
    if (dry) return;
    if (rad <= 0) { Fill(r, c); return; }
    rt->FillRoundedRectangle(D2D1::RoundedRect(DR(r), rad, rad), B(c));
}
void Gfx::StrokeRound(const Rc& r, float rad, Col c, float width) {
    if (dry) return;
    const Rc s = r.in(width / 2);
    rt->DrawRoundedRectangle(D2D1::RoundedRect(DR(s), std::max(0.f, rad - width / 2), std::max(0.f, rad - width / 2)), B(c), width);
}

// Kept by its shape (paths_; the caller does not release it): a row's corners, a cover's clip are the same every frame
ID2D1PathGeometry* Gfx::RoundPath(const Rc& r, float tl, float tr, float br, float bl) {
    auto q = [](float f) { return (int)std::lround(f * 64); };
    const auto key = std::make_tuple(q(r.x), q(r.y), q(r.w), q(r.h), q(tl), q(tr), q(br), q(bl));
    auto it = paths_.find(key);
    if (it != paths_.end()) { it->second.used = frame_; return it->second.p; }
    ID2D1PathGeometry* p = nullptr;
    if (FAILED(d2d_->CreatePathGeometry(&p))) return nullptr;
    ID2D1GeometrySink* s = nullptr;
    p->Open(&s);
    auto arc = [&](float x, float y, float rad) {
        s->AddArc(D2D1::ArcSegment(D2D1::Point2F(x, y), D2D1::SizeF(rad, rad), 0, D2D1_SWEEP_DIRECTION_CLOCKWISE, D2D1_ARC_SIZE_SMALL));
    };
    s->BeginFigure(D2D1::Point2F(r.x + tl, r.y), D2D1_FIGURE_BEGIN_FILLED);
    s->AddLine(D2D1::Point2F(r.r() - tr, r.y)); if (tr > 0) arc(r.r(), r.y + tr, tr);
    s->AddLine(D2D1::Point2F(r.r(), r.b() - br)); if (br > 0) arc(r.r() - br, r.b(), br);
    s->AddLine(D2D1::Point2F(r.x + bl, r.b())); if (bl > 0) arc(r.x, r.b() - bl, bl);
    s->AddLine(D2D1::Point2F(r.x, r.y + tl)); if (tl > 0) arc(r.x + tl, r.y, tl);
    s->EndFigure(D2D1_FIGURE_END_CLOSED);
    s->Close();
    s->Release();
    paths_[key] = CachedPath{p, frame_};
    return p;
}
void Gfx::FillRound4(const Rc& r, float tl, float tr, float br, float bl, Col c) {
    if (dry) return;
    ID2D1PathGeometry* p = RoundPath(r, tl, tr, br, bl);
    if (!p) return;
    rt->FillGeometry(p, B(c));
}

// The brush for a set of stops is made once per target and kept (grads_, released with the target); each draw only moves its
// end points. A target that has seen very many sets (plates of many names) starts the cache over.
void Gfx::GradStops(const Rc& r, float rad, const std::vector<std::pair<float, Col>>& stops, float x0, float y0, float x1, float y1, float opacity) {
    if (dry) return;
    std::vector<float> key;
    for (auto& s : stops) key.insert(key.end(), {s.first, s.second.r, s.second.g, s.second.b, s.second.a});
    auto it = grads_.find(key);
    if (it == grads_.end()) {
        if (grads_.size() >= 192) { for (auto& gb : grads_) SafeRelease(gb.second); grads_.clear(); }
        std::vector<D2D1_GRADIENT_STOP> st;
        for (auto& s : stops) st.push_back(D2D1::GradientStop(s.first, DC(s.second)));
        ID2D1GradientStopCollection* coll = nullptr;
        if (FAILED(rt->CreateGradientStopCollection(st.data(), (UINT32)st.size(), &coll))) return;
        ID2D1LinearGradientBrush* made = nullptr;
        rt->CreateLinearGradientBrush(D2D1::LinearGradientBrushProperties(D2D1::Point2F(0, 0), D2D1::Point2F(1, 0)), coll, &made);
        coll->Release();
        if (!made) return;
        it = grads_.emplace(key, made).first;
    }
    ID2D1LinearGradientBrush* lb = it->second;
    lb->SetStartPoint(D2D1::Point2F(r.x + x0 * r.w, r.y + y0 * r.h));
    lb->SetEndPoint(D2D1::Point2F(r.x + x1 * r.w, r.y + y1 * r.h));
    lb->SetOpacity(std::max(0.f, std::min(1.f, opacity)));  // set on every draw: the brush is shared by every draw of these stops
    if (rad > 0) rt->FillRoundedRectangle(D2D1::RoundedRect(DR(r), rad, rad), lb);
    else rt->FillRectangle(DR(r), lb);
}
void Gfx::Grad(const Rc& r, float rad, Col a, Col b, float x0, float y0, float x1, float y1) { GradStops(r, rad, {{0.f, a}, {1.f, b}}, x0, y0, x1, y1); }

void Gfx::Line(float x1, float y1, float x2, float y2, Col c, float width) { if (!dry) rt->DrawLine(D2D1::Point2F(x1, y1), D2D1::Point2F(x2, y2), B(c), width); }
void Gfx::Circle(float cx, float cy, float rad, Col c) { if (!dry) rt->FillEllipse(D2D1::Ellipse(D2D1::Point2F(cx, cy), rad, rad), B(c)); }
void Gfx::Ring(float cx, float cy, float rad, Col c, float width) { if (!dry) rt->DrawEllipse(D2D1::Ellipse(D2D1::Point2F(cx, cy), rad, rad), B(c), width); }

// A soft shadow without effects: rounded rectangles growing outwards, fading. A step per 1.5 device pixels of blur (10 to
// 48), so a deep shadow (a sheet's) shows no bands.
// A soft shadow as CSS draws one (a Gaussian of `blur` / 2 standard deviation across the offset shape's edge): n rounded
// rectangles of one alpha, grown and shrunk round the shape by the Gaussian's quantiles (a logistic approximation), so
// the shadow fades from the shape's edge on both sides instead of standing flat under it. Deep inside it reaches 0.73 of
// `alpha` (the level the parts' alphas were tuned to).
// `covered`: a layer shrunk by at least its offset lies inside r (the shape eroded by |e| and moved by dy <= |e| stays in it),
// where the opaque shape drawn next hides it: not filled (half of a deep shadow's layers, on every frame).
void Gfx::Shadow(const Rc& r, float rad, float blur, float alpha, float dy, bool covered) {
    if (dry || alpha <= 0) return;
    const int n = std::min(72, std::max(16, (int)std::ceil(blur * scale * 2)));  // fine steps: no band at the outer edge
    const float peak = std::min(0.95f, alpha * 0.73f), sigma = blur / 2;
    const float a = 1 - std::pow(1 - peak, 1.f / n);  // n layers of `a` add up to the peak
    for (int i = n; i >= 1; --i) {
        const float t = (i - 0.5f) / n;
        const float e = sigma * 0.5513f * std::log(t / (1 - t));  // the t quantile of a logistic with the Gaussian's spread
        const float w = r.w + 2 * e, h = r.h + 2 * e;
        if (w <= 0.5f || h <= 0.5f) continue;
        if (covered && e <= -std::fabs(dy) - 1) continue;  // hidden under the shape (a DIP inside: its edge pixels stay)
        FillRound(Rc{r.x - e, r.y - e + dy, w, h}, std::max(0.f, rad + e), Alpha(pal.black, a));
    }
}

// ---- text ----------------------------------------------------------------------------------------------------------------
IDWriteTextFormat* Gfx::Format(FontFace f, float size, int weight) {
    const auto key = std::make_tuple((int)f, (int)std::lround(size * 100), weight);
    auto it = formats_.find(key);
    if (it != formats_.end()) return it->second;
    IDWriteTextFormat* fmt = nullptr;
    dw->CreateTextFormat(families_[f].c_str(), nullptr, (DWRITE_FONT_WEIGHT)weight, DWRITE_FONT_STYLE_NORMAL, stretch_[f], size, L"en-us", &fmt);
    formats_[key] = fmt;
    return fmt;
}

IDWriteTextLayout* Gfx::Layout(const wstring& t, float w, float h, const TS& s) {
    IDWriteTextFormat* fmt = Format(s.face, s.size, s.weight);
    if (!fmt) return nullptr;
    IDWriteTextLayout* l = nullptr;
    if (FAILED(dw->CreateTextLayout(t.c_str(), (UINT32)t.size(), fmt, std::max(1.f, w), std::max(1.f, h), &l))) return nullptr;
    l->SetTextAlignment(s.align == 1 ? DWRITE_TEXT_ALIGNMENT_CENTER : s.align == 2 ? DWRITE_TEXT_ALIGNMENT_TRAILING : DWRITE_TEXT_ALIGNMENT_LEADING);
    l->SetParagraphAlignment(s.valign == 1 ? DWRITE_PARAGRAPH_ALIGNMENT_CENTER : s.valign == 2 ? DWRITE_PARAGRAPH_ALIGNMENT_FAR : DWRITE_PARAGRAPH_ALIGNMENT_NEAR);
    l->SetWordWrapping(s.wrap ? DWRITE_WORD_WRAPPING_WRAP : DWRITE_WORD_WRAPPING_NO_WRAP);
    if (!s.wrap || s.trim) {
        IDWriteInlineObject*& sign = signs_[fmt];  // one ellipsis sign per text format, kept
        if (!sign) dw->CreateEllipsisTrimmingSign(fmt, &sign);
        DWRITE_TRIMMING tr{s.wrap ? DWRITE_TRIMMING_GRANULARITY_WORD : DWRITE_TRIMMING_GRANULARITY_CHARACTER, 0, 0};
        l->SetTrimming(&tr, sign);
    }
    const DWRITE_TEXT_RANGE all{0, (UINT32)t.size()};
    if (s.spacing != 0) {
        IDWriteTextLayout1* l1 = nullptr;
        if (SUCCEEDED(l->QueryInterface(__uuidof(IDWriteTextLayout1), (void**)&l1))) { l1->SetCharacterSpacing(0, s.spacing, 0, all); l1->Release(); }
    }
    if (s.lineH > 0) l->SetLineSpacing(DWRITE_LINE_SPACING_METHOD_UNIFORM, s.lineH, s.lineH * 0.8f);
    if (s.underline) l->SetUnderline(TRUE, all);
    return l;
}

// Wrapped text whose last line is a single word (an orphan: "...on Direct3D / 12.") is wrapped again at the widest width
// that brings a second word down to it without adding a line; the height does not change, other text is left alone.
// Returns how far the narrower layout moves (centred and right-aligned text keep their place).
static float Balance(IDWriteTextLayout* l, const wstring& t, float w, const TS& s) {
    if (!s.wrap || t.find(L'\n') != wstring::npos) return 0;
    UINT32 n = 0;
    l->GetLineMetrics(nullptr, 0, &n);
    if (n < 2 || n > 6) return 0;
    DWRITE_TEXT_METRICS m{}; l->GetMetrics(&m);
    if (m.height > l->GetMaxHeight() + 0.5f) return 0;  // cut off (an ellipsis): left as it is
    std::vector<DWRITE_LINE_METRICS> lm;
    // the layout at its current width: same line count, last line of two words or more, no word broken at its hyphen
    auto good = [&]() {
        UINT32 k = 0;
        l->GetLineMetrics(nullptr, 0, &k);
        if (k != n) return false;
        lm.resize(k);
        l->GetLineMetrics(lm.data(), k, &k);
        UINT32 start = 0;
        for (UINT32 i = 0; i + 1 < k; ++i) {
            start += lm[i].length;
            if (!lm[i].trailingWhitespaceLength && t[start - 1] == L'-') return false;
        }
        // two words: a space, or NoWidow's no-break space (its words kept together are already two)
        return t.substr(start, lm[k - 1].length - lm[k - 1].trailingWhitespaceLength).find_first_of(L"  ") != wstring::npos;
    };
    if (good()) return 0;
    float lo = w * (1 - 0.5f / n), hi = w;
    l->SetMaxWidth(lo);
    if (!good()) { l->SetMaxWidth(w); return 0; }
    for (int i = 0; i < 8; ++i) {  // the widest width that is still good
        const float mid = (lo + hi) / 2;
        l->SetMaxWidth(mid);
        if (good()) lo = mid; else hi = mid;
    }
    l->SetMaxWidth(lo);
    const float slack = w - lo;
    return s.align == 1 ? slack / 2 : s.align == 2 ? slack : 0;
}

// A layout made once for its text, style (not its colour: the brush is given when it is drawn) and box, balanced when asked
// (Balance), with its metrics; kept in layouts_ (Trim lets go of those not used for a while)
const Gfx::CachedLayout* Gfx::Cached(const wstring& t, float w, float h, const TS& s, bool balance) {
    auto q = [](float f) { return (int)std::lround(std::min(f, 1e6f) * 64); };
    const int flags = (int)s.face | s.align << 3 | s.valign << 5 | (s.wrap ? 1 << 7 : 0) | (s.trim ? 1 << 8 : 0) | (s.underline ? 1 << 9 : 0) | (balance ? 1 << 10 : 0) |
                      s.weight << 12;
    const auto key = std::make_tuple(t, q(w), q(h), (int)std::lround(s.size * 100), flags, (int)std::lround(s.spacing * 100), (int)std::lround(s.lineH * 100));
    auto it = layouts_.find(key);
    if (it != layouts_.end()) { it->second.used = frame_; return &it->second; }
    CachedLayout c;
    c.l = Layout(t, w, h, s);
    if (!c.l) return nullptr;
    if (balance) c.dx = Balance(c.l, t, w, s);
    c.l->GetMetrics(&c.m);
    c.used = frame_;
    return &(layouts_[key] = c);
}

// Display text (the Display face: page titles, the welcome, a plate's initials, the mark; any text of 20 DIPs or more) in
// grayscale antialiasing on screen too, as WinUI and Apple draw it: ClearType's colour fringes read as a halo at that size.
// Body text keeps the system's choice (ClearType where it is on). Offscreen frames are grayscale throughout (TargetReady).
static bool DisplayText(const TS& s) { return s.face == F_DISPLAY || s.size >= 20; }
float Gfx::Text(const wstring& t, const Rc& r, const TS& s) {
    if (t.empty()) return 0;
    const CachedLayout* c = Cached(t, r.w, std::max(r.h, 1.f), s, true);
    if (!c) return 0;
    if (!dry) {
        const bool gray = !offscreen && DisplayText(s);
        if (gray) rt->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
        rt->DrawTextLayout(D2D1::Point2F(r.x + c->dx, r.y), c->l, B(s.color), D2D1_DRAW_TEXT_OPTIONS_NONE);
        if (gray) rt->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_DEFAULT);
    }
    return c->m.height;
}
float Gfx::TextW(const wstring& t, const TS& s) {
    if (t.empty()) return 0;
    TS n = s; n.wrap = false; n.align = 0; n.valign = 0; n.trim = false; n.underline = false; n.lineH = 0;
    const CachedLayout* c = Cached(t, 100000, 1000, n, false);
    return c ? c->m.widthIncludingTrailingWhitespace : 0;
}
float Gfx::TextH(const wstring& t, float width, const TS& s) {
    if (t.empty()) return 0;
    TS n = s; n.valign = 0; n.underline = false;
    const CachedLayout* c = Cached(t, width, 100000, n, false);
    return c ? c->m.height : 0;
}
void Gfx::TextEnd(const wstring& t, float width, const TS& s, float& x, float& y) {
    x = y = 0;
    if (t.empty()) return;
    TS n = s; n.valign = 0; n.align = 0;
    const CachedLayout* c = Cached(t, width, 100000, n, true);  // balanced: where Text draws it
    if (!c) return;
    DWRITE_HIT_TEST_METRICS m{};
    FLOAT px = 0, py = 0;
    c->l->HitTestTextPosition((UINT32)t.size() - 1, TRUE, &px, &py, &m);
    x = px; y = m.top;
}
void Gfx::Icon(wchar_t glyph, float cx, float cy, float size, Col c) {
    TS s; s.face = F_ICON; s.size = size; s.color = c; s.align = 1; s.valign = 1;
    // icons in grayscale: ClearType's coloured fringes show on their fine strokes (the chip's pins); text keeps ClearType
    const D2D1_TEXT_ANTIALIAS_MODE keep = rt ? rt->GetTextAntialiasMode() : D2D1_TEXT_ANTIALIAS_MODE_DEFAULT;
    if (rt) rt->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
    Text(wstring(1, glyph), Rc{cx - size, cy - size, size * 2, size * 2}, s);
    if (rt) rt->SetTextAntialiasMode(keep);
}

// ---- images ---------------------------------------------------------------------------------------------------------------
ID2D1Bitmap* Gfx::Bitmap(const ImageP& img) {
    if (!img || img->w <= 0 || img->h <= 0) return nullptr;
    auto it = bitmaps_.find(img.get());
    if (it != bitmaps_.end()) {
        const std::weak_ptr<const Image>& w = it->second.img;
        if (!w.owner_before(img) && !img.owner_before(w)) { it->second.used = frame_; return it->second.b; }
        SafeRelease(it->second.b); SafeRelease(it->second.shadow);  // another image at a gone one's address
        bitmaps_.erase(it);
    }
    if (bitmaps_.size() >= 128)  // many kept: those this frame has not drawn go (made again if they are drawn again)
        for (auto k = bitmaps_.begin(); k != bitmaps_.end();) {
            if (k->second.used != frame_) { SafeRelease(k->second.b); SafeRelease(k->second.shadow); k = bitmaps_.erase(k); }
            else ++k;
        }
    ID2D1Bitmap* b = nullptr;
    rt->CreateBitmap(D2D1::SizeU(img->w, img->h), img->px.data(), img->w * 4,
                     D2D1::BitmapProperties(D2D1::PixelFormat(DXGI_FORMAT_B8G8R8A8_UNORM, D2D1_ALPHA_MODE_PREMULTIPLIED), 96, 96), &b);
    bitmaps_[img.get()] = Bmp{img, b, frame_};
    return b;
}
static void DrawBmp(ID2D1RenderTarget* rt, ID2D1DeviceContext* dc, ID2D1Bitmap* b, const D2D1_RECT_F& dst, const D2D1_RECT_F& src, float opacity) {
    if (dc) dc->DrawBitmap(b, dst, opacity, D2D1_INTERPOLATION_MODE_HIGH_QUALITY_CUBIC, src, nullptr);
    else rt->DrawBitmap(b, dst, opacity, D2D1_BITMAP_INTERPOLATION_MODE_LINEAR, src);
}
void Gfx::ImageCover(const ImageP& img, const Rc& r, float rad, float focusY, float opacity, bool topOnly) {
    if (dry) return;
    ID2D1Bitmap* b = Bitmap(img);
    if (!b) return;
    const float k = std::max(r.w / img->w, r.h / img->h);
    const float sw = r.w / k, sh = r.h / k;
    const float sx = (img->w - sw) / 2, sy = (img->h - sh) * focusY;
    if (rad > 0) PushRoundClip(r, topOnly ? -rad : rad);
    DrawBmp(rt, dc_, b, DR(r), D2D1::RectF(sx, sy, sx + sw, sy + sh), opacity);
    if (rad > 0) PopRoundClip();
}
void Gfx::ImageContain(const ImageP& img, const Rc& box, int ax, int ay, float opacity, float shadow) {
    if (dry) return;
    ID2D1Bitmap* b = Bitmap(img);
    if (!b) return;
    const float k = std::min(box.w / img->w, box.h / img->h);
    const float w = img->w * k, h = img->h * k;
    const float x = box.x + (box.w - w) * ax / 2, y = box.y + (box.h - h) * ay / 2;
    // a soft drop shadow under a logo (CSS drop-shadow): the bitmap's alpha, blurred, in black. Its own effect, made once
    // with the bitmap and its output cached (Bmp::shadow): its values are set again only when they change (a fade, a size)
    auto it = bitmaps_.find(img.get());
    if (shadow > 0 && dc_ && k > 0 && it != bitmaps_.end()) {
        Bmp& bm = it->second;
        if (!bm.shadow && SUCCEEDED(dc_->CreateEffect(CLSID_D2D1Shadow, &bm.shadow))) {
            bm.shadow->SetInput(0, b);
            bm.shadow->SetValue(D2D1_PROPERTY_CACHED, TRUE);
        }
        if (ID2D1Effect* fx = bm.shadow) {
            const float sd = shadow / 2 / k, a = 0.75f * opacity;
            if (sd != bm.shadowSd) { fx->SetValue(D2D1_SHADOW_PROP_BLUR_STANDARD_DEVIATION, sd); bm.shadowSd = sd; }
            if (a != bm.shadowA) { fx->SetValue(D2D1_SHADOW_PROP_COLOR, D2D1::Vector4F(pal.artShade.r, pal.artShade.g, pal.artShade.b, a)); bm.shadowA = a; }
            D2D1_MATRIX_3X2_F old; rt->GetTransform(&old);
            rt->SetTransform(D2D1::Matrix3x2F::Scale(k, k) * D2D1::Matrix3x2F::Translation(x, y + shadow / 4) * old);
            dc_->DrawImage(fx, D2D1_INTERPOLATION_MODE_LINEAR);
            rt->SetTransform(old);
        }
    }
    DrawBmp(rt, dc_, b, D2D1::RectF(x, y, x + w, y + h), D2D1::RectF(0, 0, (float)img->w, (float)img->h), opacity);
}

void Gfx::ImageNearest(const ImageP& img, const Rc& r, float rad, float opacity) {
    if (dry) return;
    ID2D1Bitmap* b = Bitmap(img);
    if (!b) return;
    const D2D1_RECT_F src = D2D1::RectF(0, 0, (float)img->w, (float)img->h);
    if (rad > 0) PushRoundClip(r, rad);
    if (dc_) dc_->DrawBitmap(b, DR(r), opacity, D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR, src, nullptr);
    else rt->DrawBitmap(b, DR(r), opacity, D2D1_BITMAP_INTERPOLATION_MODE_NEAREST_NEIGHBOR, src);
    if (rad > 0) PopRoundClip();
}

// Made once per text, style, box and lead, and kept with the other layouts (SHIP-BAR 1: a Settings status row drawn on
// every frame of a scroll or a fade lays nothing out again); only its lead's brush is given again as it is drawn (the
// target's lead brush: a layout outlives a target).
float Gfx::TextLead(const wstring& t, size_t leadLen, Col leadC, int leadWeight, const Rc& r, const TS& s) {
    if (t.empty()) return 0;
    const DWRITE_TEXT_RANGE lead{0, (UINT32)std::min(leadLen, t.size())};
    auto q = [](float f) { return (int)std::lround(std::min(f, 1e6f) * 64); };
    const int flags = (int)s.face | s.align << 3 | s.valign << 5 | (s.wrap ? 1 << 7 : 0) | (s.trim ? 1 << 8 : 0) | (s.underline ? 1 << 9 : 0) | 1 << 10 | 1 << 11 |
                      s.weight << 12;
    const wstring k = t + L'\x1' + std::to_wstring(lead.length) + L',' + std::to_wstring(leadWeight);  // the lead in the key
    const auto key = std::make_tuple(k, q(r.w), q(std::max(r.h, 1.f)), (int)std::lround(s.size * 100), flags, (int)std::lround(s.spacing * 100), (int)std::lround(s.lineH * 100));
    auto it = layouts_.find(key);
    if (it == layouts_.end()) {
        CachedLayout c;
        c.l = Layout(t, r.w, std::max(r.h, 1.f), s);
        if (!c.l) return 0;
        c.dx = Balance(c.l, t, r.w, s);
        c.lead = true;
        c.l->SetFontWeight((DWRITE_FONT_WEIGHT)leadWeight, lead);
        c.l->GetMetrics(&c.m);
        it = layouts_.emplace(key, c).first;
    }
    CachedLayout& c = it->second;
    c.used = frame_;
    if (!dry && leadBrush_) { leadBrush_->SetColor(DC(leadC)); c.l->SetDrawingEffect(leadBrush_, lead); }
    if (!dry) {
        const bool gray = !offscreen && DisplayText(s);  // as Text
        if (gray) rt->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
        rt->DrawTextLayout(D2D1::Point2F(r.x + c.dx, r.y), c.l, B(s.color), D2D1_DRAW_TEXT_OPTIONS_NONE);
        if (gray) rt->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_DEFAULT);
    }
    return c.m.height;
}

void Gfx::Glow(float cx, float cy, float rx, float ry, Col c) {
    if (dry) return;
    // a brush per colour, made once per target (each draw moves its centre and radii)
    auto q = [](float f) { return (int)std::lround(f * 1000); };
    ID2D1RadialGradientBrush*& rb = glows_[std::make_tuple(q(c.r), q(c.g), q(c.b), q(c.a))];
    if (!rb) {
        const D2D1_GRADIENT_STOP st[3] = {D2D1::GradientStop(0, DC(c)), D2D1::GradientStop(0.45f, DC(Col{c.r, c.g, c.b, c.a * 0.45f})), D2D1::GradientStop(1, DC(Col{c.r, c.g, c.b, 0}))};
        ID2D1GradientStopCollection* coll = nullptr;
        if (FAILED(rt->CreateGradientStopCollection(st, 3, &coll))) return;
        rt->CreateRadialGradientBrush(D2D1::RadialGradientBrushProperties(D2D1::Point2F(cx, cy), D2D1::Point2F(0, 0), rx, ry), coll, &rb);
        coll->Release();
        if (!rb) return;
    }
    rb->SetCenter(D2D1::Point2F(cx, cy));
    rb->SetRadiusX(rx);
    rb->SetRadiusY(ry);
    rt->FillEllipse(D2D1::Ellipse(D2D1::Point2F(cx, cy), rx, ry), rb);
}

void Gfx::PushClip(const Rc& r) { if (!dry) { rt->PushAxisAlignedClip(DR(r), D2D1_ANTIALIAS_MODE_PER_PRIMITIVE); ++clips_; } }
void Gfx::PopClip() { if (!dry) { rt->PopAxisAlignedClip(); --clips_; } }
// rad < 0: only the top corners are rounded
void Gfx::PushRoundClip(const Rc& r, float rad) {
    if (dry) return;
    const float a = std::fabs(rad);
    ID2D1PathGeometry* p = RoundPath(r, a, a, rad < 0 ? 0 : a, rad < 0 ? 0 : a);
    // no layer object: Direct2D (Windows 8 and later) manages the layer's resources itself, nothing is made per frame
    rt->PushLayer(D2D1::LayerParameters(D2D1::InfiniteRect(), p, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE), nullptr);
    layers_.push_back(nullptr);
}
void Gfx::PopRoundClip() {
    if (dry) return;
    if (layers_.empty()) return;
    rt->PopLayer();
    SafeRelease(layers_.back());
    layers_.pop_back();
}
// A layer at `opacity`. It starts from what is under it (INITIALIZE_FOR_CLEARTYPE), so ClearType text in the live window
// stays ClearType while it fades in.
void Gfx::PushFade(float opacity) {
    if (dry) return;
    rt->PushLayer(D2D1::LayerParameters(D2D1::InfiniteRect(), nullptr, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE, D2D1::IdentityMatrix(), std::min(1.f, std::max(0.f, opacity)),
                                        nullptr, D2D1_LAYER_OPTIONS_INITIALIZE_FOR_CLEARTYPE),
                  nullptr);
    layers_.push_back(nullptr);
}
void Gfx::PopFade() { PopRoundClip(); }

void Gfx::PushMask(const Rc& r, float rad, float opacity) {
    ID2D1PathGeometry* p = rad > 0 ? RoundPath(r, rad, rad, rad, rad) : nullptr;
    rt->PushLayer(D2D1::LayerParameters(rad > 0 ? D2D1::InfiniteRect() : DR(r), p, D2D1_ANTIALIAS_MODE_PER_PRIMITIVE, D2D1::IdentityMatrix(),
                                        std::min(1.f, std::max(0.f, opacity)), nullptr, D2D1_LAYER_OPTIONS_NONE),
                  nullptr);
    layers_.push_back(nullptr);
}

// The backdrop blur (the floating capsule, toasts, the scrim under a sheet): the target's pixels under r, plus three
// standard deviations round it for the blur to sample, are copied into a bitmap (CopyFromRenderTarget needs no clip or
// layer open), blurred with Direct2D's Gaussian blur (hard borders: the window's edge does not fade in) and drawn back
// through a rounded mask. Live, the target is the swap chain's back buffer; offscreen, the WIC bitmap: the same code.
bool Gfx::Backdrop(const Rc& r, float rad, float blur, float opacity) {
    if (dry) return true;
    if (!dc_ || !layers_.empty() || clips_ > 0 || blur <= 0.05f || opacity <= 0) return false;
    const D2D1_SIZE_U ps = rt->GetPixelSize();
    const float m = blur * 3, wx = r.x + ox_, wy = r.y + oy_;  // window DIPs
    const UINT32 x0 = (UINT32)std::max(0.f, std::floor((wx - m) * scale)), y0 = (UINT32)std::max(0.f, std::floor((wy - m) * scale));
    const UINT32 x1 = (UINT32)std::min((float)ps.width, std::ceil((wx + r.w + m) * scale)), y1 = (UINT32)std::min((float)ps.height, std::ceil((wy + r.h + m) * scale));
    if (x1 <= x0 + 1 || y1 <= y0 + 1) return false;
    float dpx = 96, dpy = 96;
    rt->GetDpi(&dpx, &dpy);
    // the copies and the blur effect are kept per target: two copies, one per size in use, so the unsaved capsule and a toast
    // on screen together (each its own size) reuse theirs every frame instead of making one again twice a frame
    int slot = -1;
    for (int i = 0; i < 2 && slot < 0; ++i)
        if (backdropBmp_[i]) {
            const D2D1_SIZE_U have = backdropBmp_[i]->GetPixelSize();
            if (have.width == x1 - x0 && have.height == y1 - y0) slot = i;
        }
    if (slot < 0) {
        slot = !backdropBmp_[0] ? 0 : !backdropBmp_[1] ? 1 : backdropNext_;
        SafeRelease(backdropBmp_[slot]);
    }
    backdropNext_ = 1 - slot;  // the copy not used now is the one a new size replaces
    backdropUsed_[slot] = frame_;
    const D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(D2D1_BITMAP_OPTIONS_NONE, rt->GetPixelFormat(), dpx, dpy);
    bool ok = backdropBmp_[slot] || SUCCEEDED(dc_->CreateBitmap(D2D1::SizeU(x1 - x0, y1 - y0), nullptr, 0, bp, &backdropBmp_[slot]));
    if (ok) {
        const D2D1_POINT_2U at = D2D1::Point2U(0, 0);
        const D2D1_RECT_U src = D2D1::RectU(x0, y0, x1, y1);
        ok = SUCCEEDED(backdropBmp_[slot]->CopyFromRenderTarget(&at, rt, &src)) && (blur_ || SUCCEEDED(dc_->CreateEffect(CLSID_D2D1GaussianBlur, &blur_)));
    }
    ID2D1Effect* fx = blur_;
    ID2D1Bitmap1* bmp = backdropBmp_[slot];
    if (ok) {
        fx->SetInput(0, bmp);
        fx->SetValue(D2D1_GAUSSIANBLUR_PROP_STANDARD_DEVIATION, blur);
        fx->SetValue(D2D1_GAUSSIANBLUR_PROP_BORDER_MODE, D2D1_BORDER_MODE_HARD);
        fx->SetValue(D2D1_GAUSSIANBLUR_PROP_OPTIMIZATION, D2D1_GAUSSIANBLUR_OPTIMIZATION_QUALITY);
        PushMask(r, rad, opacity);
        dc_->DrawImage(fx, D2D1::Point2F(x0 / scale - ox_, y0 / scale - oy_), D2D1_INTERPOLATION_MODE_LINEAR);
        PopRoundClip();
    }
    return ok;
}

// The whole frame as drawn so far kept as the scrim's picture (Scrim, once its entrance ends), and drawn back while the page
// behind the sheet shows the same thing (`key`): a dialog's hover or a first-run line no longer copies and blurs the window.
bool Gfx::DrawFrozen(unsigned long long key) {
    if (dry) return true;
    if (!frozen_ || frozenKey_ != key || !dc_ || !layers_.empty() || clips_ > 0) return false;
    const D2D1_SIZE_U ps = rt->GetPixelSize(), have = frozen_->GetPixelSize();
    if (have.width != ps.width || have.height != ps.height) return false;
    frozenUsed_ = frame_;
    D2D1_MATRIX_3X2_F old; rt->GetTransform(&old);
    rt->SetTransform(D2D1::Matrix3x2F::Identity());
    dc_->DrawBitmap(frozen_, D2D1::RectF(0, 0, ps.width / scale, ps.height / scale), 1, D2D1_INTERPOLATION_MODE_NEAREST_NEIGHBOR, nullptr, nullptr);
    rt->SetTransform(old);
    return true;
}
void Gfx::Freeze(unsigned long long key) {
    if (dry || !dc_ || !layers_.empty() || clips_ > 0) return;
    const D2D1_SIZE_U ps = rt->GetPixelSize();
    if (frozen_) {
        const D2D1_SIZE_U have = frozen_->GetPixelSize();
        if (have.width != ps.width || have.height != ps.height) SafeRelease(frozen_);
    }
    float dpx = 96, dpy = 96;
    rt->GetDpi(&dpx, &dpy);
    const D2D1_BITMAP_PROPERTIES1 bp = D2D1::BitmapProperties1(D2D1_BITMAP_OPTIONS_NONE, rt->GetPixelFormat(), dpx, dpy);
    if (!frozen_ && FAILED(dc_->CreateBitmap(ps, nullptr, 0, bp, &frozen_))) { frozen_ = nullptr; return; }
    const D2D1_POINT_2U at = D2D1::Point2U(0, 0);
    const D2D1_RECT_U src = D2D1::RectU(0, 0, ps.width, ps.height);
    frozenKey_ = SUCCEEDED(frozen_->CopyFromRenderTarget(&at, rt, &src)) ? key : 0;
    frozenUsed_ = frame_;
}

}  // namespace wg::app::lui
