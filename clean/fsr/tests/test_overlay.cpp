// SPDX-License-Identifier: MIT
// Overlay test (overlay.h): a D3D12 swap chain on a hidden window per back-buffer format; the overlay is fed statuses
// directly (wgwin::OverlayStatus) and draws in Present; the presented buffer is read back (flip-sequential keeps it)
// and checked for the panel (darker than the grey clear colour) and the state colour (green ACTIVE, red OFF), also
// after a resize. Writes overlay_<format>_<state>.ppm previews (sRGB-encoded 8-bit) next to the executable.
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "overlay.h"

namespace wronggpu {
extern const char* const kLogTag = "test_overlay";
}

namespace {

int g_failures = 0;
#define CHECK(c, ...)                         \
    do {                                      \
        if (!(c)) {                           \
            std::printf("FAIL: " __VA_ARGS__); \
            std::printf("\n");                \
            ++g_failures;                     \
        }                                     \
    } while (0)

float Half(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? 65504.0f : std::ldexp(float(m | 1024), int(e) - 25);
    return s ? -v : v;
}

uint16_t ToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    int e = int((x >> 23) & 255) - 127 + 15;
    uint32_t m = (x >> 13) & 1023;
    if (e <= 0) return uint16_t(s);
    if (e >= 31) return uint16_t(s | 0x7C00);
    return uint16_t(s | (uint32_t(e) << 10) | m);
}

struct Rgb {
    float r, g, b;
};

// Pixel as display-referred 0..1 sRGB-ish values (scRGB: linear / (203/80) -> sRGB; PQ is not used in this test).
Rgb Decode(DXGI_FORMAT f, const uint8_t* p) {
    if (f == DXGI_FORMAT_R8G8B8A8_UNORM) return {p[0] / 255.f, p[1] / 255.f, p[2] / 255.f};
    if (f == DXGI_FORMAT_B8G8R8A8_UNORM) return {p[2] / 255.f, p[1] / 255.f, p[0] / 255.f};
    if (f == DXGI_FORMAT_R10G10B10A2_UNORM) {
        uint32_t v;
        std::memcpy(&v, p, 4);
        return {(v & 1023) / 1023.f, ((v >> 10) & 1023) / 1023.f, ((v >> 20) & 1023) / 1023.f};
    }
    uint16_t h[3];
    std::memcpy(h, p, 6);
    auto enc = [](float lin) {
        lin = lin / (203.0f / 80.0f);
        lin = lin < 0 ? 0 : lin > 1 ? 1 : lin;
        return lin <= 0.0031308f ? lin * 12.92f : 1.055f * std::pow(lin, 1 / 2.4f) - 0.055f;
    };
    return {enc(Half(h[0])), enc(Half(h[1])), enc(Half(h[2]))};
}

UINT Bpp(DXGI_FORMAT f) { return f == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8 : 4; }

// Copy path (overlay.cpp, Renderer): when set, a COMPUTE queue gets an (empty) submission on the presenting thread right
// before Present, as FSR 3's frame-interpolation swap chain does with its own queue; the overlay then copies its panel on
// that queue.
ID3D12InfoQueue* g_info = nullptr;  // D3D12 debug layer messages (when the Graphics Tools are installed)

void DumpMessages(const char* where) {
    if (!g_info) return;
    const UINT64 n = g_info->GetNumStoredMessages();
    for (UINT64 i = 0; i < n && i < 20; ++i) {
        SIZE_T size = 0;
        g_info->GetMessage(i, nullptr, &size);
        std::vector<char> buf(size);
        auto* m = reinterpret_cast<D3D12_MESSAGE*>(buf.data());
        if (SUCCEEDED(g_info->GetMessage(i, m, &size)) && m->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
            std::printf("D3D12 [%s] %s\n", where, m->pDescription);
    }
    g_info->ClearStoredMessages();
}

ID3D12CommandQueue* g_extraQueue = nullptr;
ID3D12CommandAllocator* g_extraAlloc = nullptr;
ID3D12GraphicsCommandList* g_extraList = nullptr;
ID3D12Fence* g_extraFence = nullptr;
UINT64 g_extraValue = 0;

struct Gpu {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    UINT64 value = 0;
    void Wait() {
        queue->Signal(fence, ++value);
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(value, e);
        WaitForSingleObject(e, 5000);
        CloseHandle(e);
    }
};

void Barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
    l->ResourceBarrier(1, &x);
}

// Clears the current back buffer to mid grey, presents (the overlay draws in Present), reads the presented buffer back.
std::vector<Rgb> PresentAndRead(Gpu& g, IDXGISwapChain3* sc, DXGI_FORMAT f, UINT w, UINT h, ID3D12DescriptorHeap* rtvHeap) {
    const UINT index = sc->GetCurrentBackBufferIndex();
    ID3D12Resource* back = nullptr;
    sc->GetBuffer(index, IID_PPV_ARGS(&back));
    g.alloc->Reset();
    g.list->Reset(g.alloc, nullptr);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    g.device->CreateRenderTargetView(back, nullptr, rtv);
    // The clear stands in for an upscale dispatch: timed (nested, as FSR 3.0 calling its upscaler) for the upscaler time.
    wgwin::OverlayGpuTimer(g.list, false);
    wgwin::OverlayGpuTimer(g.list, false);
    Barrier(g.list, back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    float grey = f == DXGI_FORMAT_R16G16B16A16_FLOAT ? 0.214f * (203.0f / 80.0f) : 0.5f;  // sRGB 0.5 in each encoding
    const float clear[4] = {grey, grey, grey, 1};
    g.list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    Barrier(g.list, back, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    wgwin::OverlayGpuTimer(g.list, true);
    wgwin::OverlayGpuTimer(g.list, true);
    g.list->Close();
    ID3D12CommandList* lists[] = {g.list};
    g.queue->ExecuteCommandLists(1, lists);
    if (g_extraQueue) {
        ID3D12CommandList* extra[] = {g_extraList};
        g_extraQueue->ExecuteCommandLists(1, extra);
    }
    DumpMessages("before Present");
    sc->Present(0, 0);
    DumpMessages("Present");
    g.Wait();
    if (FAILED(g.device->GetDeviceRemovedReason())) std::printf("device removed: 0x%08lx\n", static_cast<unsigned long>(g.device->GetDeviceRemovedReason()));
    if (g_extraQueue) {  // the overlay's copy on the compute queue
        g_extraQueue->Signal(g_extraFence, ++g_extraValue);
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        g_extraFence->SetEventOnCompletion(g_extraValue, e);
        WaitForSingleObject(e, 5000);
        CloseHandle(e);
    }

    const UINT pitch = (w * Bpp(f) + 255u) & ~255u;
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = UINT64(pitch) * h;
    rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* rb = nullptr;
    g.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&rb));
    g.alloc->Reset();
    g.list->Reset(g.alloc, nullptr);
    Barrier(g.list, back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = rb;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint = {f, w, h, 1, pitch};
    src.pResource = back;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    g.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Barrier(g.list, back, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
    g.list->Close();
    g.queue->ExecuteCommandLists(1, lists);
    g.Wait();
    std::vector<Rgb> px(size_t(w) * h);
    uint8_t* data = nullptr;
    rb->Map(0, nullptr, reinterpret_cast<void**>(&data));
    for (UINT y = 0; y < h; ++y)
        for (UINT x = 0; x < w; ++x) px[size_t(y) * w + x] = Decode(f, data + size_t(y) * pitch + size_t(x) * Bpp(f));
    rb->Unmap(0, nullptr);
    rb->Release();
    back->Release();
    return px;
}

// At least `frames` frames and 0.45 s of them (the overlay fades in over 0.2 s); the last one read back.
std::vector<Rgb> PresentFrames(Gpu& g, IDXGISwapChain3* sc, DXGI_FORMAT f, UINT w, UINT h, ID3D12DescriptorHeap* rtvHeap, int frames) {
    std::vector<Rgb> px;
    const DWORD start = GetTickCount();
    for (int i = 0; i < frames || GetTickCount() - start < 450; ++i) px = PresentAndRead(g, sc, f, w, h, rtvHeap);
    return px;
}

void WritePpm(const std::string& name, const std::vector<Rgb>& px, UINT w, UINT h) {
    FILE* f = std::fopen(name.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6 %u %u 255\n", w, h);
    for (const Rgb& p : px) {
        const unsigned char c[3] = {(unsigned char)(std::fmin(1.f, std::fmax(0.f, p.r)) * 255 + 0.5f),
                                    (unsigned char)(std::fmin(1.f, std::fmax(0.f, p.g)) * 255 + 0.5f),
                                    (unsigned char)(std::fmin(1.f, std::fmax(0.f, p.b)) * 255 + 0.5f)};
        std::fwrite(c, 1, 3, f);
    }
    std::fclose(f);
}

struct Counts {
    int panel = 0, green = 0, red = 0;
};

// Within the top-left 900 x 300 pixels: panel = clearly darker than the grey; green / red text pixels.
Counts Count(const std::vector<Rgb>& px, UINT w, UINT h) {
    Counts c;
    for (UINT y = 0; y < std::min(h, 300u); ++y)
        for (UINT x = 0; x < std::min(w, 900u); ++x) {
            const Rgb& p = px[size_t(y) * w + x];
            if (p.r < 0.36f && p.g < 0.36f && p.b < 0.36f) ++c.panel;
            if (p.g > 0.6f && p.g > p.r + 0.25f && p.g > p.b + 0.2f) ++c.green;
            if (p.r > 0.6f && p.r > p.g + 0.3f && p.r > p.b + 0.3f) ++c.red;
        }
    return c;
}

const char* Name(DXGI_FORMAT f) {
    return f == DXGI_FORMAT_R8G8B8A8_UNORM ? "rgba8" : f == DXGI_FORMAT_B8G8R8A8_UNORM ? "bgra8" : f == DXGI_FORMAT_R10G10B10A2_UNORM ? "rgb10a2" : "rgba16f";
}

void Status(wgfsr::UpscaleState state, void* device, const char* reason) {
    wgfsr::UpscaleStatus s;
    s.state = state;
    s.upscaler = "WrongGPU network";
    s.reason = reason;
    s.renderWidth = 1707;
    s.renderHeight = 960;
    s.outputWidth = 2560;
    s.outputHeight = 1440;
    s.device = device;
    wgwin::OverlayStatus(s);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    wgfsr::Config config;  // overlay = on
    config.overlayPosition = "top-left";  // the checks look at the top-left corner
    wgwin::ConfigureOverlay(config);
    Gpu g;
    {
        ID3D12Debug* debug = nullptr;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))) && debug) {
            debug->EnableDebugLayer();
            debug->Release();
        }
    }
    if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&g.device)))) {
        std::printf("SKIP: no D3D12 device\n");
        return 0;
    }
    if (SUCCEEDED(g.device->QueryInterface(IID_PPV_ARGS(&g_info))) && g_info) std::printf("D3D12 debug layer on\n");
    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
    g.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue));
    g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.alloc));
    g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.alloc, nullptr, IID_PPV_ARGS(&g.list));
    g.list->Close();
    g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence));
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    g.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtvHeap));

    // The first status with a device installs the hooks (as in a game, before the game's swap chains present).
    Status(wgfsr::UpscaleState::Network, g.device, "");

    IDXGIFactory2* factory = nullptr;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"WrongGPUOverlayTest";
    RegisterClassExW(&wc);
    const DXGI_FORMAT formats[] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM,
                                   DXGI_FORMAT_R16G16B16A16_FLOAT};
    for (DXGI_FORMAT f : formats) {
        HWND wnd = CreateWindowExW(0, wc.lpszClassName, L"overlay test", WS_OVERLAPPEDWINDOW, 0, 0, 1280, 720, nullptr, nullptr,
                                   wc.hInstance, nullptr);
        UINT w = 1280, h = 720;
        DXGI_SWAP_CHAIN_DESC1 d{};
        d.Width = w;
        d.Height = h;
        d.Format = f;
        d.SampleDesc.Count = 1;
        d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.BufferCount = 3;
        d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        IDXGISwapChain1* sc1 = nullptr;
        if (FAILED(factory->CreateSwapChainForHwnd(g.queue, wnd, &d, nullptr, nullptr, &sc1))) {
            CHECK(false, "%s: CreateSwapChainForHwnd failed", Name(f));
            DestroyWindow(wnd);
            continue;
        }
        IDXGISwapChain3* sc = nullptr;
        sc1->QueryInterface(IID_PPV_ARGS(&sc));
        if (f == DXGI_FORMAT_R16G16B16A16_FLOAT) sc->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709);
        if (f == DXGI_FORMAT_R10G10B10A2_UNORM) sc->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);  // SDR (an HDR display would make it PQ)

        Status(wgfsr::UpscaleState::Network, g.device, "");
        std::vector<Rgb> px;
        px = PresentFrames(g, sc, f, w, h, rtvHeap, 4);  // a few frames: atlas upload, frame rate
        Counts c = Count(px, w, h);
        WritePpm(std::string("overlay_") + Name(f) + "_active.ppm", px, w, h);
        std::printf("%s ACTIVE: panel %d, green %d, red %d\n", Name(f), c.panel, c.green, c.red);
        CHECK(c.panel > 5000 && c.green > 200 && c.red < 400 /* the wordmark's red bar only */, "%s: ACTIVE panel not drawn as expected", Name(f));
        const double upscalerMs = wgwin::OverlayUpscalerMs();
        std::printf("%s upscaler time: %.4f ms\n", Name(f), upscalerMs);
        CHECK(upscalerMs > 0 && upscalerMs < 50, "%s: no upscaler time measured", Name(f));
        CHECK(px[size_t(h - 10) * w + (w - 10)].g > 0.4f && px[size_t(h - 10) * w + (w - 10)].g < 0.6f, "%s: image outside the panel changed", Name(f));

        // A failed frame must immediately replace a still-recent ACTIVE state.
        Status(wgfsr::UpscaleState::Failed, g.device, "AMD fallback is unavailable");
        px = PresentFrames(g, sc, f, w, h, rtvHeap, 3);
        c = Count(px, w, h);
        WritePpm(std::string("overlay_") + Name(f) + "_failed.ppm", px, w, h);
        std::printf("%s FAILED: panel %d, green %d, red %d\n", Name(f), c.panel, c.green, c.red);
        CHECK(c.panel > 5000 && c.red > 200 && c.green == 0, "%s: failed frame retained ACTIVE state", Name(f));
        Status(wgfsr::UpscaleState::Network, g.device, "");
        px = PresentFrames(g, sc, f, w, h, rtvHeap, 3);
        c = Count(px, w, h);
        CHECK(c.panel > 5000 && c.green > 200, "%s: successful frame did not recover from failure", Name(f));

        // The same frame state change: AMD with a reason -> red.
        Sleep(1100);  // the network state ages out (1 s)
        Status(wgfsr::UpscaleState::Amd, g.device, "network runtime unavailable: no converted model at C:\\example\\model.wgcm");
        px = PresentFrames(g, sc, f, w, h, rtvHeap, 3);
        c = Count(px, w, h);
        WritePpm(std::string("overlay_") + Name(f) + "_off.ppm", px, w, h);
        std::printf("%s OFF: panel %d, green %d, red %d\n", Name(f), c.panel, c.green, c.red);
        CHECK(c.panel > 5000 && c.red > 200 && c.green == 0, "%s: OFF panel not drawn as expected", Name(f));

        // Resize: the overlay forgets the old buffers and draws on the new ones.
        w = 1920;
        h = 1080;
        g.Wait();
        CHECK(SUCCEEDED(sc->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0)), "%s: ResizeBuffers failed", Name(f));
        Status(wgfsr::UpscaleState::Network, g.device, "");
        px = PresentFrames(g, sc, f, w, h, rtvHeap, 3);
        c = Count(px, w, h);
        std::printf("%s after resize: panel %d, green %d\n", Name(f), c.panel, c.green);
        CHECK(c.panel > 5000 && c.green > 200, "%s: nothing drawn after the resize", Name(f));
        Sleep(1100);
        sc->Release();
        sc1->Release();
        DestroyWindow(wnd);
    }
    // Panel then badge (overlay = on): a status every frame for 12 s; after 10 s the panel turns into the badge.
    {
        const DXGI_FORMAT f = DXGI_FORMAT_R8G8B8A8_UNORM;
        HWND wnd = CreateWindowExW(0, wc.lpszClassName, L"overlay test", WS_OVERLAPPEDWINDOW, 0, 0, 1280, 720, nullptr, nullptr,
                                   wc.hInstance, nullptr);
        DXGI_SWAP_CHAIN_DESC1 d{};
        d.Width = 1280;
        d.Height = 720;
        d.Format = f;
        d.SampleDesc.Count = 1;
        d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.BufferCount = 3;
        d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        IDXGISwapChain1* sc1 = nullptr;
        if (SUCCEEDED(factory->CreateSwapChainForHwnd(g.queue, wnd, &d, nullptr, nullptr, &sc1))) {
            IDXGISwapChain3* sc = nullptr;
            sc1->QueryInterface(IID_PPV_ARGS(&sc));
            Sleep(1100);  // earlier states age out: the panel starts here
            std::vector<Rgb> px;
            const DWORD start = GetTickCount();
            while (GetTickCount() - start < 12500) {
                Status(wgfsr::UpscaleState::Network, g.device, "");
                px = PresentAndRead(g, sc, f, 1280, 720, rtvHeap);
            }
            const Counts c = Count(px, 1280, 720);
            WritePpm("overlay_after_panel.ppm", px, 1280, 720);
            std::printf("after the panel (badge): panel %d, green %d, red %d\n", c.panel, c.green, c.red);
            CHECK(c.panel > 1500 && c.panel < 15000 && c.green > 40, "after the panel: badge not drawn");
            // The overlay key hides it; shown again, it starts with the full panel for 10 s again (not the badge).
            wgwin::OverlayToggle();
            for (DWORD t0 = GetTickCount(); GetTickCount() - t0 < 1000;) {
                Status(wgfsr::UpscaleState::Network, g.device, "");
                px = PresentAndRead(g, sc, f, 1280, 720, rtvHeap);
            }
            const Counts hidden = Count(px, 1280, 720);
            wgwin::OverlayToggle();
            for (DWORD t0 = GetTickCount(); GetTickCount() - t0 < 1000;) {
                Status(wgfsr::UpscaleState::Network, g.device, "");
                px = PresentAndRead(g, sc, f, 1280, 720, rtvHeap);
            }
            const Counts again = Count(px, 1280, 720);
            WritePpm("overlay_shown_again.ppm", px, 1280, 720);
            std::printf("hidden: panel %d; shown again: panel %d, green %d\n", hidden.panel, again.panel, again.green);
            CHECK(hidden.panel == 0, "overlay key: not hidden");
            CHECK(again.panel > 5000 && again.green > 200, "overlay key: shown again without the full panel");
            sc->Release();
            sc1->Release();
        }
        DestroyWindow(wnd);
    }
    // Badge (overlay = badge): the compact pill only.
    {
        wgfsr::Config badge;
        badge.overlay = "badge";
        badge.hosted = true;
        badge.overlayPosition = "top-left";
        wgwin::ConfigureOverlay(badge);
        const DXGI_FORMAT f = DXGI_FORMAT_R8G8B8A8_UNORM;
        HWND wnd = CreateWindowExW(0, wc.lpszClassName, L"overlay test", WS_OVERLAPPEDWINDOW, 0, 0, 1280, 720, nullptr, nullptr,
                                   wc.hInstance, nullptr);
        DXGI_SWAP_CHAIN_DESC1 d{};
        d.Width = 1280;
        d.Height = 720;
        d.Format = f;
        d.SampleDesc.Count = 1;
        d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.BufferCount = 3;
        d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        IDXGISwapChain1* sc1 = nullptr;
        if (SUCCEEDED(factory->CreateSwapChainForHwnd(g.queue, wnd, &d, nullptr, nullptr, &sc1))) {
            IDXGISwapChain3* sc = nullptr;
            sc1->QueryInterface(IID_PPV_ARGS(&sc));
            Status(wgfsr::UpscaleState::Network, g.device, "");
            std::vector<Rgb> px;
            px = PresentFrames(g, sc, f, 1280, 720, rtvHeap, 4);
            const Counts c = Count(px, 1280, 720);
            WritePpm("overlay_badge_active.ppm", px, 1280, 720);
            std::printf("badge ACTIVE: panel %d, green %d, red %d\n", c.panel, c.green, c.red);
            CHECK(c.panel > 1500 && c.green > 40 && c.red < 400 /* the wordmark's red bar only */, "badge: not drawn as expected");
            wgfsr::UpscaleStatus amd;
            amd.state = wgfsr::UpscaleState::Amd;
            amd.upscaler = "4.1.1";  // public AMD provider lists can return the numeric version only
            amd.device = g.device;
            wgwin::OverlayStatus(amd);
            px = PresentFrames(g, sc, f, 1280, 720, rtvHeap, 4);
            const Counts switched = Count(px, 1280, 720);
            WritePpm("overlay_badge_hosted_amd.ppm", px, 1280, 720);
            std::printf("hosted AMD badge: panel %d, green %d, red %d\n", switched.panel, switched.green, switched.red);
            CHECK(switched.panel > 1500 && switched.green < 20, "hosted badge: stale network state after AMD dispatch");
            sc->Release();
            sc1->Release();
        }
        DestroyWindow(wnd);
        wgwin::ConfigureOverlay(config);
    }
    // Copy path: rgba8 with a compute-queue submission before each Present.
    {
        D3D12_COMMAND_QUEUE_DESC cd{D3D12_COMMAND_LIST_TYPE_COMPUTE};
        g.device->CreateCommandQueue(&cd, IID_PPV_ARGS(&g_extraQueue));
        g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&g_extraAlloc));
        g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, g_extraAlloc, nullptr, IID_PPV_ARGS(&g_extraList));
        g_extraList->Close();
        g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g_extraFence));
        const DXGI_FORMAT f = DXGI_FORMAT_R8G8B8A8_UNORM;
        HWND wnd = CreateWindowExW(0, wc.lpszClassName, L"overlay test", WS_OVERLAPPEDWINDOW, 0, 0, 1280, 720, nullptr, nullptr,
                                   wc.hInstance, nullptr);
        DXGI_SWAP_CHAIN_DESC1 d{};
        d.Width = 1280;
        d.Height = 720;
        d.Format = f;
        d.SampleDesc.Count = 1;
        d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.BufferCount = 3;
        d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        IDXGISwapChain1* sc1 = nullptr;
        if (SUCCEEDED(factory->CreateSwapChainForHwnd(g.queue, wnd, &d, nullptr, nullptr, &sc1))) {
            IDXGISwapChain3* sc = nullptr;
            sc1->QueryInterface(IID_PPV_ARGS(&sc));
            Status(wgfsr::UpscaleState::Network, g.device, "");
            std::vector<Rgb> px;
            px = PresentFrames(g, sc, f, 1280, 720, rtvHeap, 6);
            const Counts c = Count(px, 1280, 720);
            WritePpm("overlay_copy_path_active.ppm", px, 1280, 720);
            std::printf("copy path ACTIVE: panel %d, green %d, red %d\n", c.panel, c.green, c.red);
            CHECK(c.panel > 5000 && c.green > 200 && c.red < 400 /* the wordmark's red bar only */, "copy path: panel not drawn as expected");
            sc->Release();
            sc1->Release();
        } else {
            CHECK(false, "copy path: CreateSwapChainForHwnd failed");
        }
        DestroyWindow(wnd);
        g_extraQueue = nullptr;  // (released with the process)
    }
    factory->Release();
    std::printf(g_failures ? "%d FAILURES\n" : "overlay tests passed\n", g_failures);
    return g_failures ? 1 : 0;
}
