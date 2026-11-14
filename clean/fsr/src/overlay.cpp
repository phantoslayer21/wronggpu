// SPDX-License-Identifier: MIT
// In-game overlay (overlay.h). Written for WrongGPU.
#include "overlay.h"

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "log_win.h"
#include "win_util.h"

namespace wgwin {
namespace {

using wronggpu::Log;

// ---------------------------------------------------------------------------------------------------------------
// Settings of this DLL

struct Settings {
    bool hosted = false; // a host switches whole contexts: the most recent successful dispatch wins
    int mode = 1;   // 0 off, 1 on (panel, then badge), 2 badge
    int key = 0x2D; // VK_INSERT
    int compareKey = 0x24;  // VK_HOME; 0: off
    std::wstring keyName = L"Insert", compareName = L"Home";
    bool right = false, bottom = true;  // corner
};
Settings g_settings;

int64_t Now();

// "Insert", "Page Up", "F9", "Key 0x2D" for a configured key name.
std::wstring KeyLabel(const std::string& name) {
    static const struct {
        const char* name;
        const wchar_t* label;
    } kLabels[] = {{"insert", L"Insert"}, {"home", L"Home"}, {"end", L"End"}, {"delete", L"Delete"}, {"pageup", L"Page Up"},
                   {"pagedown", L"Page Down"}, {"pause", L"Pause"}, {"scrolllock", L"Scroll Lock"}};
    for (const auto& k : kLabels)
        if (name == k.name) return k.label;
    std::wstring w(name.begin(), name.end());
    if (!w.empty() && w[0] == L'f') w[0] = L'F';
    if (w.size() > 2 && w[0] == L'0') w = L"Key " + w;
    return w;
}

// ---------------------------------------------------------------------------------------------------------------
// Status shared by WrongGPU's DLLs of the process (named mapping; same layout in every DLL of one build)

constexpr uint32_t kMagic = 0x564F4757;  // "WGOV"
constexpr uint32_t kLayout = 3;
constexpr int kStates = 5;           // wgfsr::UpscaleState values
constexpr unsigned kTimerSlots = 16; // upscaler time: timestamp pairs in flight (see GpuTimer)

struct SharedStatus {
    uint32_t magic;
    uint32_t layout;
    volatile LONG owner;      // 1 once a DLL has claimed the overlay
    SRWLOCK lock;             // guards everything below
    int64_t lastQpc[kStates]; // QueryPerformanceCounter of the last dispatch in each UpscaleState
    uint64_t frames[kStates];
    int lastState;            // of the last report
    uint32_t renderW, renderH, outW, outH;  // of the last report
    char upscaler[64];
    char reason[256];
    double upscaleMs;         // upscaler GPU time, smoothed
    int64_t upscaleQpc;       // when last measured
    int upscaleSkip;          // measurements still to drop after the upscaler changed (they timed the previous one)
    bool upscaleFresh;        // the next measurement starts the average again
    volatile LONG compare;    // 1: the player switched to AMD's FSR (compare_key)
    volatile LONG timerThread;  // thread id of the dispatch being timed (nested dispatches, also in other DLLs, are not)
};

SharedStatus* Shared() {
    static SharedStatus* block = [] () -> SharedStatus* {
        wchar_t name[96];
        std::swprintf(name, 96, L"Local\\WrongGPU-FSR-Overlay-%lu", GetCurrentProcessId());
        HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(SharedStatus), name);
        const bool created = h && GetLastError() != ERROR_ALREADY_EXISTS;
        auto* s = h ? static_cast<SharedStatus*>(MapViewOfFile(h, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(SharedStatus))) : nullptr;
        if (!s) {
            Log::Get().Write("overlay: shared status block not available (error %lu); overlay off", GetLastError());
            return nullptr;
        }
        // The first DLL initialises the block (zeroed by the system); the magic is written last.
        if (created) {
            InitializeSRWLock(&s->lock);
            s->layout = kLayout;
            InterlockedExchange(reinterpret_cast<volatile LONG*>(&s->magic), LONG(kMagic));
        } else {
            for (int i = 0; i < 1000 && s->magic != kMagic; ++i) Sleep(1);
        }
        if (s->magic != kMagic || s->layout != kLayout) {
            Log::Get().Write("overlay: shared status block of another WrongGPU build (layout %u); overlay off in this DLL", s->layout);
            return nullptr;
        }
        return s;  // handle and view stay open for the life of the process
    }();
    return block;
}

int64_t Now() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return t.QuadPart;
}

double Seconds(int64_t ticks) {
    static const double freq = [] {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return double(f.QuadPart);
    }();
    return double(ticks) / freq;
}

void CopyText(char* dst, size_t size, const std::string& src) {
    std::snprintf(dst, size, "%s", src.c_str());
}

// ---------------------------------------------------------------------------------------------------------------
// Swap chain vtable hooks

constexpr int kSlotPresent = 8;          // IDXGISwapChain::Present
constexpr int kSlotResizeBuffers = 13;   // IDXGISwapChain::ResizeBuffers
constexpr int kSlotPresent1 = 22;        // IDXGISwapChain1::Present1
constexpr int kSlotSetColorSpace1 = 38;  // IDXGISwapChain3::SetColorSpace1
constexpr int kSlotResizeBuffers1 = 39;  // IDXGISwapChain3::ResizeBuffers1
constexpr int kSlots = 40;

constexpr int kSlotExecuteCommandLists = 10;  // ID3D12CommandQueue::ExecuteCommandLists

struct PatchedVtbl {
    void** vtbl = nullptr;
    void* orig[kSlots] = {};
};
struct Family {
    PatchedVtbl entries[4];
    std::atomic<int> count{0};
};
Family g_swapChainVt, g_queueVt;

void* Original(Family& family, const void* self, int slot) {
    void** vt = *static_cast<void** const*>(self);
    const int n = family.count.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i)
        if (family.entries[i].vtbl == vt) return family.entries[i].orig[slot];
    return n ? family.entries[0].orig[slot] : nullptr;
}
void* Original(const void* self, int slot) { return Original(g_swapChainVt, self, slot); }

// A D3D12 swap chain does not give out its queue (GetDevice answers E_NOINTERFACE), so the overlay remembers the queues
// the game submits to (ExecuteCommandLists). The present queue is taken to be the one submitted to last on the presenting
// thread. FSR 3's frame-interpolation swap chain, for one, presents from a copy or compute queue of its own (see
// Renderer for how the overlay gets onto the back buffer then).
struct QueueSeen {
    ID3D12CommandQueue* queue = nullptr;  // referenced
    bool direct = false;
    DWORD thread = 0;
    int64_t when = 0;
};
QueueSeen g_queues[8];
SRWLOCK g_queueLock = SRWLOCK_INIT;

using PfnPresent = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT);
using PfnPresent1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain1*, UINT, UINT, const DXGI_PRESENT_PARAMETERS*);
using PfnResizeBuffers = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain*, UINT, UINT, UINT, DXGI_FORMAT, UINT);
using PfnResizeBuffers1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain3*, UINT, UINT, UINT, DXGI_FORMAT, UINT, const UINT*,
                                                      IUnknown* const*);
using PfnSetColorSpace1 = HRESULT(STDMETHODCALLTYPE*)(IDXGISwapChain3*, DXGI_COLOR_SPACE_TYPE);

void NoteQueue(ID3D12CommandQueue* q) {
    const DWORD thread = GetCurrentThreadId();
    const int64_t now = Now();
    AcquireSRWLockExclusive(&g_queueLock);
    QueueSeen* slot = nullptr;
    QueueSeen* oldest = &g_queues[0];
    for (auto& e : g_queues) {
        if (e.queue == q) {
            slot = &e;
            break;
        }
        if (e.when < oldest->when) oldest = &e;
    }
    if (!slot) {
        slot = oldest;
        if (slot->queue) slot->queue->Release();
        slot->queue = q;
        const D3D12_COMMAND_QUEUE_DESC d = q->GetDesc();
        slot->direct = d.Type == D3D12_COMMAND_LIST_TYPE_DIRECT;
        q->AddRef();
    }
    slot->thread = thread;
    slot->when = now;
    ReleaseSRWLockExclusive(&g_queueLock);
}

// The queue (any type) submitted to last on this thread: the queue the swap chain presents from (referenced), or null.
ID3D12CommandQueue* PickPresentQueue() {
    const DWORD thread = GetCurrentThreadId();
    AcquireSRWLockShared(&g_queueLock);
    const QueueSeen* p = nullptr;
    for (const auto& e : g_queues)
        if (e.queue && e.thread == thread && (!p || e.when > p->when)) p = &e;
    ID3D12CommandQueue* q = p ? p->queue : nullptr;
    if (q) q->AddRef();
    ReleaseSRWLockShared(&g_queueLock);
    return q;
}

using PfnExecute = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, UINT, ID3D12CommandList* const*);

// A swap chain's back buffers may only be written on the swap chain's own queue (anything else removes the device), and
// DXGI does not give that queue out. But during the original Present, DXGI submits and signals on exactly that queue, so
// the hooks learn it there: the queue is known from the swap chain's second Present on.
thread_local IDXGISwapChain* t_inPresent = nullptr;  // the swap chain whose original Present runs on this thread

struct SwapChainQueue {
    IDXGISwapChain* sc = nullptr;     // identity only
    ID3D12CommandQueue* queue = nullptr;  // referenced
    int64_t when = 0;
};
SwapChainQueue g_scQueues[8];
SRWLOCK g_scLock = SRWLOCK_INIT;

void ProbeNote(ID3D12CommandQueue* q, const char*) {
    IDXGISwapChain* sc = t_inPresent;
    if (!sc) return;
    AcquireSRWLockExclusive(&g_scLock);
    SwapChainQueue* slot = nullptr;
    SwapChainQueue* oldest = &g_scQueues[0];
    for (auto& e : g_scQueues) {
        if (e.sc == sc) {
            slot = &e;
            break;
        }
        if (e.when < oldest->when) oldest = &e;
    }
    if (!slot) slot = oldest;
    if (slot->sc != sc || slot->queue != q) {
        if (slot->queue) slot->queue->Release();
        slot->sc = sc;
        slot->queue = q;
        q->AddRef();
    }
    slot->when = Now();
    ReleaseSRWLockExclusive(&g_scLock);
}

// The queue learnt for this swap chain (referenced), or null.
ID3D12CommandQueue* SwapChainQueueOf(IDXGISwapChain* sc) {
    AcquireSRWLockShared(&g_scLock);
    ID3D12CommandQueue* q = nullptr;
    for (const auto& e : g_scQueues)
        if (e.sc == sc) q = e.queue;
    if (q) q->AddRef();
    ReleaseSRWLockShared(&g_scLock);
    return q;
}

void ForgetSwapChainQueue(IDXGISwapChain* sc) {
    AcquireSRWLockExclusive(&g_scLock);
    for (auto& e : g_scQueues)
        if (e.sc == sc) {
            if (e.queue) e.queue->Release();
            e = SwapChainQueue{};
        }
    ReleaseSRWLockExclusive(&g_scLock);
}

constexpr int kSlotSignal = 14, kSlotWait = 15;
using PfnSignal = HRESULT(STDMETHODCALLTYPE*)(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
HRESULT STDMETHODCALLTYPE HSignal(ID3D12CommandQueue* q, ID3D12Fence* f, UINT64 v) {
    ProbeNote(q, "Signal");
    return reinterpret_cast<PfnSignal>(Original(g_queueVt, q, kSlotSignal))(q, f, v);
}
HRESULT STDMETHODCALLTYPE HWait(ID3D12CommandQueue* q, ID3D12Fence* f, UINT64 v) {
    ProbeNote(q, "Wait");
    return reinterpret_cast<PfnSignal>(Original(g_queueVt, q, kSlotWait))(q, f, v);
}

void STDMETHODCALLTYPE HExecuteCommandLists(ID3D12CommandQueue* q, UINT n, ID3D12CommandList* const* lists) {
    ProbeNote(q, "ExecuteCommandLists");
    NoteQueue(q);
    reinterpret_cast<PfnExecute>(Original(g_queueVt, q, kSlotExecuteCommandLists))(q, n, lists);
}

// ---------------------------------------------------------------------------------------------------------------
// Renderer (one per swap chain)

constexpr const char kShader[] = R"HLSL(
// pos: x0 y0 x1 y1 (pixels). extra.x: 0 glyph, 1 rounded box; boxes: extra.y corner radius, extra.z edge softness
// (pixels; shadows and glows), extra.w inset of the box within the quad (room for the soft edge), uv: multipliers
// (alpha top, alpha bottom, colour top, colour bottom) for a vertical gradient.
struct Quad { float4 pos; float4 uv; float4 color; float4 extra; };
StructuredBuffer<Quad> quads : register(t1);
Texture2D<float> font : register(t0);
SamplerState samp : register(s0);
cbuffer C : register(b0) { float2 invSize; uint mode; float nits; };
struct V {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    nointerpolation float4 color : COLOR0;
    float2 local : TEXCOORD1;
    nointerpolation float4 extra : TEXCOORD2;
    nointerpolation float2 size : TEXCOORD3;
    nointerpolation float4 grad : TEXCOORD4;
};
V vs(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
    const float2 corner[6] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(0, 1), float2(1, 0), float2(1, 1) };
    Quad q = quads[iid];
    float2 t = corner[vid % 6];
    float2 p = lerp(q.pos.xy, q.pos.zw, t);
    V o;
    o.pos = float4(p.x * invSize.x * 2 - 1, 1 - p.y * invSize.y * 2, 0, 1);
    o.uv = lerp(q.uv.xy, q.uv.zw, t);
    o.color = q.color;
    o.size = q.pos.zw - q.pos.xy;
    o.local = t * o.size;
    o.extra = q.extra;
    o.grad = q.uv;
    return o;
}
float3 SrgbToLinear(float3 c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }
float3 Pq(float3 nitsIn) {
    float3 y = pow(saturate(nitsIn / 10000), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * y) / (1 + 18.6875 * y), 78.84375);
}
float4 ps(V i) : SV_Target {
    float a;
    float4 c = i.color;
    if (i.extra.x > 0.5) {  // rounded box: signed distance; one pixel of anti-aliasing, or a soft edge
        float2 h = i.size * 0.5 - i.extra.w;
        float r = min(i.extra.y, min(h.x, h.y));
        float2 d2 = abs(i.local - i.size * 0.5) - (h - r);
        float d = length(max(d2, 0)) + min(max(d2.x, d2.y), 0) - r;
        float soft = i.extra.z;
        a = soft > 1 ? 1 - smoothstep(-soft * 0.5, soft * 0.5, d) : saturate(0.5 - d);
        float t = saturate(i.local.y / i.size.y);
        a *= lerp(i.grad.x, i.grad.y, t);
        c.rgb *= lerp(i.grad.z, i.grad.w, t);
    } else {
        a = font.Sample(samp, i.uv);
    }
    c.a *= a;
    if (mode == 1) {
        c.rgb = SrgbToLinear(c.rgb) * (nits / 80);
    } else if (mode == 2) {
        float3 lin = SrgbToLinear(c.rgb);
        float3 r2020 = float3(dot(float3(0.6274, 0.3293, 0.0433), lin), dot(float3(0.0691, 0.9195, 0.0114), lin),
                              dot(float3(0.0164, 0.0880, 0.8956), lin));
        c.rgb = Pq(r2020 * nits);
    }
    return c;
}
)HLSL";

struct Quad {
    float pos[4];
    float uv[4];
    float color[4];
    float extra[4];  // [0]: 0 glyph, 1 rounded box; boxes: [1] corner radius, [2] edge softness, [3] inset (pixels)
};
constexpr UINT kMaxQuads = 768;
constexpr UINT kMaxBuffers = 8;

// Characters in the atlas: printable ASCII and a few typographic ones.
constexpr wchar_t kExtraChars[] = {0x00D7 /* multiplication sign */, 0x2192 /* right arrow */, 0x00B7 /* middle dot */, 0x2022 /* bullet */};
constexpr int kChars = 95 + int(sizeof kExtraChars / sizeof kExtraChars[0]);

int CharIndex(wchar_t c) {
    if (c >= 32 && c < 127) return c - 32;
    for (int i = 0; i < int(sizeof kExtraChars / sizeof kExtraChars[0]); ++i)
        if (kExtraChars[i] == c) return 95 + i;
    return '?' - 32;
}

struct Glyph {
    float u0 = 0, v0 = 0, u1 = 0, v1 = 0;
    int w = 0, h = 0;     // cell in the atlas (pixels)
    int offset = 0;       // cell's left edge relative to the pen position
    int advance = 0;
};
struct Font {
    Glyph glyph[kChars];
    int height = 0;  // line height (pixels)
    int ascent = 0;
};
enum FontId { kBody = 0, kBold = 1, kSmall = 2, kFonts = 3 };
struct Glyphs {
    Font font[kFonts];
    int texW = 0, texH = 0;
    std::vector<uint8_t> pixels;  // R8 coverage, texW x texH
};

// Rasterises the atlas with GDI from Segoe UI (a Windows font): body, bold title and small text, sized for `scale`
// (1 = 1080p). Glyphs are packed in rows; each keeps its ABC metrics so text is proportionally spaced.
bool RasteriseGlyphs(float scale, Glyphs& g) {
    struct Spec {
        int px, weight;
    } specs[kFonts] = {{int(std::lround(15.0f * scale)), FW_NORMAL}, {int(std::lround(17.0f * scale)), FW_BLACK},
                       {int(std::lround(12.5f * scale)), FW_NORMAL}};
    HDC dc = CreateCompatibleDC(nullptr);
    if (!dc) return false;
    // Pass 1: metrics and packing; pass 2: drawing into the DIB.
    struct Placed {
        int x, y;
    } placed[kFonts][kChars];
    HFONT fonts[kFonts] = {};
    const int texW = 1024;
    int x = 0, y = 0, rowH = 0;
    bool ok = true;
    for (int f = 0; f < kFonts && ok; ++f) {
        fonts[f] = CreateFontW(-specs[f].px, 0, 0, 0, specs[f].weight, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_TT_PRECIS,
                               CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, VARIABLE_PITCH | FF_SWISS, L"Segoe UI");
        if (!fonts[f]) {
            ok = false;
            break;
        }
        SelectObject(dc, fonts[f]);
        TEXTMETRICW tm{};
        GetTextMetricsW(dc, &tm);
        Font& font = g.font[f];
        font.height = tm.tmHeight;
        font.ascent = tm.tmAscent;
        for (int i = 0; i < kChars; ++i) {
            const wchar_t c = i < 95 ? wchar_t(32 + i) : kExtraChars[i - 95];
            ABC abc{};
            if (!GetCharABCWidthsW(dc, c, c, &abc)) {
                SIZE sz{};
                GetTextExtentPoint32W(dc, &c, 1, &sz);
                abc.abcA = 0;
                abc.abcB = UINT(sz.cx);
                abc.abcC = 0;
            }
            Glyph& gl = font.glyph[i];
            gl.w = int(abc.abcB) + 4;
            gl.h = tm.tmHeight + 2;
            gl.offset = abc.abcA - 2;
            gl.advance = abc.abcA + int(abc.abcB) + abc.abcC;
            if (x + gl.w > texW) {
                x = 0;
                y += rowH;
                rowH = 0;
            }
            placed[f][i] = {x, y};
            x += gl.w;
            rowH = std::max(rowH, gl.h);
        }
    }
    const int texH = y + rowH;
    if (ok && texH > 0 && texH <= 2048) {
        g.texW = texW;
        g.texH = texH;
        BITMAPINFO bi{};
        bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
        bi.bmiHeader.biWidth = texW;
        bi.bmiHeader.biHeight = -texH;  // top-down
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        void* bits = nullptr;
        HBITMAP bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
        ok = bmp && bits;
        if (ok) {
            HGDIOBJ oldBmp = SelectObject(dc, bmp);
            std::memset(bits, 0, size_t(texW) * texH * 4);
            SetBkMode(dc, TRANSPARENT);
            SetTextColor(dc, RGB(255, 255, 255));
            for (int f = 0; f < kFonts; ++f) {
                SelectObject(dc, fonts[f]);
                for (int i = 0; i < kChars; ++i) {
                    const wchar_t c = i < 95 ? wchar_t(32 + i) : kExtraChars[i - 95];
                    Glyph& gl = g.font[f].glyph[i];
                    const Placed pl = placed[f][i];
                    if (c != L' ') TextOutW(dc, pl.x - gl.offset, pl.y + 1, &c, 1);
                    gl.u0 = float(pl.x) / texW;
                    gl.v0 = float(pl.y) / texH;
                    gl.u1 = float(pl.x + gl.w) / texW;
                    gl.v1 = float(pl.y + gl.h) / texH;
                }
            }
            GdiFlush();
            g.pixels.resize(size_t(texW) * texH);
            const auto* px = static_cast<const uint8_t*>(bits);
            for (size_t i = 0; i < g.pixels.size(); ++i) g.pixels[i] = px[i * 4 + 1];  // green channel = coverage
            SelectObject(dc, oldBmp);
        }
        if (bmp) DeleteObject(bmp);
    } else {
        ok = false;
    }
    for (HFONT f : fonts)
        if (f) DeleteObject(f);
    DeleteDC(dc);
    return ok;
}

template <class T>
void SafeRelease(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

constexpr UINT kPanels = 3;  // copy path: panel textures in rotation

// Per swap chain. Two ways to put the overlay on the back buffer, neither ever makes the present queue wait for another
// queue:
//  - direct: the present queue is a DIRECT queue (the usual case): the text is drawn onto the back buffer on that queue,
//    right before Present.
//  - copy: the present queue is a copy or compute queue (FSR 3's frame-interpolation swap chain presents this way, after
//    copying the game's frame on that queue): the panel is drawn into a texture of the back buffer's format on a DIRECT
//    queue of the overlay's own; once the CPU sees that texture finished (fence value, no wait), a CopyTextureRegion on
//    the present queue puts it onto the back buffer right before Present. The panel is opaque in this path.
struct Renderer {
    IDXGISwapChain3* key = nullptr;  // not referenced: identity only
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;  // the present queue
    D3D12_COMMAND_LIST_TYPE queueType = D3D12_COMMAND_LIST_TYPE_DIRECT;
    bool copyPath = false;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    UINT buffers = 0, width = 0, height = 0;
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pso = nullptr;
    ID3D12DescriptorHeap* rtvHeap = nullptr;  // direct: one per back buffer; copy: one per panel texture
    ID3D12DescriptorHeap* srvHeap = nullptr;
    UINT rtvStep = 0;
    ID3D12Resource* font = nullptr;
    ID3D12Resource* fontUpload = nullptr;
    bool fontReady = false;
    Glyphs glyphs;
    ID3D12Fence* fence = nullptr;  // signalled on the present queue after each overlay submission there
    UINT64 fenceValue = 0;
    struct Slot {  // per back buffer: the overlay's command list on the present queue
        ID3D12CommandAllocator* alloc = nullptr;
        ID3D12GraphicsCommandList* list = nullptr;
        ID3D12Resource* quads = nullptr;  // direct path only
        Quad* mapped = nullptr;
        UINT64 done = 0;
    } slot[kMaxBuffers];
    // Copy path.
    ID3D12CommandQueue* ownQueue = nullptr;
    ID3D12Fence* ownFence = nullptr;
    UINT64 ownValue = 0;
    UINT panelW = 0, panelH = 0;
    struct Panel {
        ID3D12Resource* tex = nullptr;
        ID3D12CommandAllocator* alloc = nullptr;
        ID3D12GraphicsCommandList* list = nullptr;
        ID3D12Resource* quads = nullptr;
        Quad* mapped = nullptr;
        UINT64 rendered = 0;  // ownFence value of its last render
        UINT64 copied = 0;    // fence value (present queue) of the last copy that read it
        UINT64 seq = 0;       // render order
        UINT boxW = 0, boxH = 0;
    } panel[kPanels];
    UINT64 panelSeq = 0;
    float scale = 1.0f;
    int64_t lastUse = 0;
    int queueMismatch = 0;

    static void WaitFor(ID3D12Fence* f, UINT64 v) {
        if (!f || f->GetCompletedValue() >= v) return;
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (e && SUCCEEDED(f->SetEventOnCompletion(v, e))) WaitForSingleObject(e, 2000);
        if (e) CloseHandle(e);
    }
    void Release() {
        WaitFor(fence, fenceValue);
        WaitFor(ownFence, ownValue);
        for (auto& s : slot) {
            if (s.quads && s.mapped) s.quads->Unmap(0, nullptr);
            SafeRelease(s.quads);
            SafeRelease(s.list);
            SafeRelease(s.alloc);
            s.mapped = nullptr;
            s.done = 0;
        }
        for (auto& p : panel) {
            if (p.quads && p.mapped) p.quads->Unmap(0, nullptr);
            SafeRelease(p.quads);
            SafeRelease(p.list);
            SafeRelease(p.alloc);
            SafeRelease(p.tex);
            p = Panel{};
        }
        SafeRelease(ownFence);
        SafeRelease(ownQueue);
        SafeRelease(fence);
        SafeRelease(fontUpload);
        SafeRelease(font);
        SafeRelease(srvHeap);
        SafeRelease(rtvHeap);
        SafeRelease(pso);
        SafeRelease(root);
        SafeRelease(queue);
        SafeRelease(device);
        fontReady = false;
        copyPath = false;
        fenceValue = ownValue = panelSeq = 0;
    }
};

constexpr int kHistory = 240;   // frame times kept for the graph
constexpr double kFade = 0.2;   // seconds, fades in and out

struct State {
    std::mutex mutex;
    Renderer renderers[4];
    ID3DBlob* vs = nullptr;
    ID3DBlob* ps = nullptr;
    bool shadersFailed = false;
    bool disabled = false;  // after an exception or an unrecoverable error
    bool visible = true;
    bool keyDown = false;
    bool compareKeyDown = false;
    int64_t lastPresent = 0;
    double frameTime = 0;   // smoothed seconds per Present
    float history[kHistory] = {};  // seconds per Present, ring buffer
    int historyHead = 0, historyCount = 0;
    double shownFps = 0, shownMs = 0, shownUpscaleMs = 0;  // the numbers on screen, refreshed 4 times a second
    int64_t textNext = 0;
    int64_t shownAt = 0, hiddenAt = 0;  // fades: overlay key, first appearance, badge -> panel
    int lastShownState = -1;
    uint64_t presents = 0, drawn = 0, busy = 0;  // diagnostics, logged every 600 presents
    int64_t panelUntil = 0; // full panel until this time (mode "on")
    DXGI_COLOR_SPACE_TYPE colorSpace[4] = {};
    IDXGISwapChain3* colorSpaceKey[4] = {};
    std::vector<std::string> loggedOnce;
};
State* g_state = nullptr;

// The overlay key: hides or shows the overlay. Shown again, it starts with the full panel for 10 s (overlay = on), then
// the badge, as when it first appeared.
void Toggle(int64_t now) {
    g_state->visible = !g_state->visible;
    (g_state->visible ? g_state->shownAt : g_state->hiddenAt) = now;
    if (g_state->visible) g_state->panelUntil = now + int64_t(10.0 / Seconds(1));
}

void LogOnce(const std::string& what) {
    for (const auto& s : g_state->loggedOnce)
        if (s == what) return;
    g_state->loggedOnce.push_back(what);
    Log::Get().Write("overlay: %s", what.c_str());
}

bool CompileShaders() {
    if (g_state->vs && g_state->ps) return true;
    if (g_state->shadersFailed) return false;
    using PfnCompile = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR, LPCSTR, UINT,
                                        UINT, ID3DBlob**, ID3DBlob**);
    HMODULE compiler = LoadLibraryW(L"d3dcompiler_47.dll");
    auto compile = compiler ? ProcAddress<PfnCompile>(compiler, "D3DCompile") : nullptr;
    if (!compile) {
        g_state->shadersFailed = true;
        LogOnce("d3dcompiler_47.dll (part of Windows) could not be loaded; overlay off");
        return false;
    }
    for (int i = 0; i < 2; ++i) {
        ID3DBlob* code = nullptr;
        ID3DBlob* errors = nullptr;
        const HRESULT hr = compile(kShader, sizeof kShader - 1, "wronggpu_overlay", nullptr, nullptr, i ? "ps" : "vs",
                                   i ? "ps_5_0" : "vs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        if (FAILED(hr)) {
            g_state->shadersFailed = true;
            LogOnce(std::string("shader compile failed: ") + (errors ? static_cast<const char*>(errors->GetBufferPointer()) : "?"));
            SafeRelease(errors);
            return false;
        }
        SafeRelease(errors);
        (i ? g_state->ps : g_state->vs) = code;
    }
    return true;
}

int ColorModeFor(IDXGISwapChain3* sc, DXGI_FORMAT format) {
    if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) return 1;  // scRGB
    if (format != DXGI_FORMAT_R10G10B10A2_UNORM) return 0;
    for (int i = 0; i < 4; ++i)
        if (g_state->colorSpaceKey[i] == sc) return g_state->colorSpace[i] == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 ? 2 : 0;
    // Colour space set before the hooks: a 10-bit swap chain on an output in HDR mode is taken as HDR10.
    int mode = 0;
    IDXGIOutput* out = nullptr;
    if (SUCCEEDED(sc->GetContainingOutput(&out)) && out) {
        IDXGIOutput6* out6 = nullptr;
        DXGI_OUTPUT_DESC1 d{};
        if (SUCCEEDED(out->QueryInterface(IID_PPV_ARGS(&out6))) && out6 && SUCCEEDED(out6->GetDesc1(&d)))
            mode = d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 ? 2 : 0;
        SafeRelease(out6);
        out->Release();
    }
    return mode;
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition = {r, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, a, b};
    return x;
}

bool MakeList(ID3D12Device* d, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator** alloc, ID3D12GraphicsCommandList** list) {
    if (FAILED(d->CreateCommandAllocator(type, IID_PPV_ARGS(alloc))) ||
        FAILED(d->CreateCommandList(0, type, *alloc, nullptr, IID_PPV_ARGS(list))))
        return false;
    (*list)->Close();
    return true;
}

bool MakeQuads(ID3D12Device* d, ID3D12Resource** buffer, Quad** mapped) {
    D3D12_HEAP_PROPERTIES heapUpload{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC qb{};
    qb.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    qb.Width = sizeof(Quad) * kMaxQuads;
    qb.Height = qb.DepthOrArraySize = qb.MipLevels = 1;
    qb.SampleDesc.Count = 1;
    qb.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(d->CreateCommittedResource(&heapUpload, D3D12_HEAP_FLAG_NONE, &qb, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(buffer))) &&
           SUCCEEDED((*buffer)->Map(0, nullptr, reinterpret_cast<void**>(mapped)));
}

// Creates the renderer's D3D12 objects for the swap chain's current buffers and its present queue. False (logged once)
// when it cannot.
bool Create(Renderer& r, IDXGISwapChain3* sc, ID3D12CommandQueue* queue, const DXGI_SWAP_CHAIN_DESC1& desc, UINT buffers) {
    r.Release();
    r.key = sc;
    if (desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM &&
        desc.Format != DXGI_FORMAT_R10G10B10A2_UNORM && desc.Format != DXGI_FORMAT_R16G16B16A16_FLOAT) {
        LogOnce("back buffer format " + std::to_string(int(desc.Format)) + " not drawn (overlay skips this swap chain)");
        return false;
    }
    if (buffers == 0 || buffers > kMaxBuffers) {
        LogOnce("swap chain with " + std::to_string(buffers) + " buffers not drawn");
        return false;
    }
    if (!CompileShaders()) return false;
    if (FAILED(queue->GetDevice(IID_PPV_ARGS(&r.device)))) return false;
    r.queue = queue;
    queue->AddRef();
    r.queueType = queue->GetDesc().Type;
    r.copyPath = r.queueType != D3D12_COMMAND_LIST_TYPE_DIRECT;
    r.format = desc.Format;
    r.buffers = buffers;
    r.width = desc.Width;
    r.height = desc.Height;
    r.scale = desc.Height >= 1080 ? float(desc.Height) / 1080.0f : 1.0f;
    ID3D12Device* d = r.device;

    // Root signature: b0 constants, t1 quads (root SRV), t0 font (table), s0 static sampler.
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 1;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[2].DescriptorTable.NumDescriptorRanges = 1;
    params[2].DescriptorTable.pDescriptorRanges = &range;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC rootDesc{};
    rootDesc.NumParameters = 3;
    rootDesc.pParameters = params;
    rootDesc.NumStaticSamplers = 1;
    rootDesc.pStaticSamplers = &sampler;
    using PfnSerialize = HRESULT(WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
    HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
    auto serialize = d3d12 ? ProcAddress<PfnSerialize>(d3d12, "D3D12SerializeRootSignature") : nullptr;
    ID3DBlob* blob = nullptr;
    if (!serialize || FAILED(serialize(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr)) ||
        FAILED(d->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&r.root)))) {
        SafeRelease(blob);
        LogOnce("root signature failed; overlay off for this swap chain");
        return false;
    }
    SafeRelease(blob);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = r.root;
    pso.VS = {g_state->vs->GetBufferPointer(), g_state->vs->GetBufferSize()};
    pso.PS = {g_state->ps->GetBufferPointer(), g_state->ps->GetBufferSize()};
    D3D12_RENDER_TARGET_BLEND_DESC& b = pso.BlendState.RenderTarget[0];
    b.BlendEnable = TRUE;
    b.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    b.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    b.BlendOp = D3D12_BLEND_OP_ADD;
    b.SrcBlendAlpha = D3D12_BLEND_ONE;
    b.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
    b.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    b.LogicOp = D3D12_LOGIC_OP_NOOP;
    b.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.SampleMask = UINT_MAX;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = desc.Format;
    pso.SampleDesc.Count = 1;
    if (FAILED(d->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&r.pso)))) {
        LogOnce("pipeline state failed; overlay off for this swap chain");
        return false;
    }

    D3D12_DESCRIPTOR_HEAP_DESC rtv{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, r.copyPath ? kPanels : buffers, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC srv{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    if (FAILED(d->CreateDescriptorHeap(&rtv, IID_PPV_ARGS(&r.rtvHeap))) || FAILED(d->CreateDescriptorHeap(&srv, IID_PPV_ARGS(&r.srvHeap))))
        return false;
    r.rtvStep = d->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Glyph atlas: GDI rasterisation, uploaded by the first drawing command list.
    if (!RasteriseGlyphs(r.scale, r.glyphs)) {
        LogOnce("font rasterisation (GDI) failed; overlay off");
        return false;
    }
    D3D12_HEAP_PROPERTIES heapDefault{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_HEAP_PROPERTIES heapUpload{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = UINT64(r.glyphs.texW);
    tex.Height = UINT(r.glyphs.texH);
    tex.DepthOrArraySize = 1;
    tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R8_UNORM;
    tex.SampleDesc.Count = 1;
    if (FAILED(d->CreateCommittedResource(&heapDefault, D3D12_HEAP_FLAG_NONE, &tex, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                          IID_PPV_ARGS(&r.font))))
        return false;
    const UINT pitch = (UINT(r.glyphs.texW) + 255u) & ~255u;
    D3D12_RESOURCE_DESC buf{};
    buf.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buf.Width = UINT64(pitch) * UINT(r.glyphs.texH);
    buf.Height = buf.DepthOrArraySize = buf.MipLevels = 1;
    buf.SampleDesc.Count = 1;
    buf.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(d->CreateCommittedResource(&heapUpload, D3D12_HEAP_FLAG_NONE, &buf, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                          IID_PPV_ARGS(&r.fontUpload))))
        return false;
    uint8_t* up = nullptr;
    if (FAILED(r.fontUpload->Map(0, nullptr, reinterpret_cast<void**>(&up)))) return false;
    for (int y = 0; y < r.glyphs.texH; ++y)
        std::memcpy(up + size_t(y) * pitch, r.glyphs.pixels.data() + size_t(y) * r.glyphs.texW, size_t(r.glyphs.texW));
    r.fontUpload->Unmap(0, nullptr);
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R8_UNORM;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MipLevels = 1;
    d->CreateShaderResourceView(r.font, &view, r.srvHeap->GetCPUDescriptorHandleForHeapStart());

    if (FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&r.fence)))) return false;
    for (UINT i = 0; i < buffers; ++i) {
        auto& s = r.slot[i];
        if (!MakeList(d, r.queueType, &s.alloc, &s.list)) return false;
        if (!r.copyPath && !MakeQuads(d, &s.quads, &s.mapped)) return false;
    }
    if (r.copyPath) {
        D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
        if (FAILED(d->CreateCommandQueue(&qd, IID_PPV_ARGS(&r.ownQueue))) ||
            FAILED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&r.ownFence))))
            return false;
        r.panelW = std::min(desc.Width, UINT(std::lround(1500.0f * r.scale)));
        r.panelH = std::min(desc.Height, UINT(std::lround(420.0f * r.scale)));
        D3D12_RESOURCE_DESC pt = tex;
        pt.Width = r.panelW;
        pt.Height = r.panelH;
        pt.Format = desc.Format;
        pt.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        for (UINT i = 0; i < kPanels; ++i) {
            auto& p = r.panel[i];
            if (FAILED(d->CreateCommittedResource(&heapDefault, D3D12_HEAP_FLAG_NONE, &pt, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  IID_PPV_ARGS(&p.tex))) ||
                !MakeList(d, D3D12_COMMAND_LIST_TYPE_DIRECT, &p.alloc, &p.list) || !MakeQuads(d, &p.quads, &p.mapped))
                return false;
            D3D12_CPU_DESCRIPTOR_HANDLE h = r.rtvHeap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * r.rtvStep;
            d->CreateRenderTargetView(p.tex, nullptr, h);
        }
    }
    Log::Get().Write("overlay: drawing on a %ux%u swap chain (format %d, %u buffers, colour mode %d, present queue type %d: %s)",
                     r.width, r.height, int(r.format), r.buffers, ColorModeFor(sc, desc.Format), int(r.queueType),
                     r.copyPath ? "panel rendered on the overlay's own queue, copied on the present queue" : "drawn on the present queue");
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Text layout

struct Painter {
    const Glyphs& g;
    Quad* out;
    UINT count = 0;
    float alpha = 1;  // fades

    // A rounded box; `grad`: vertical gradient (alpha top, alpha bottom, colour top, colour bottom).
    void Box(float x0, float y0, float x1, float y1, float radius, const float c[4], const float grad[4] = nullptr) {
        if (count >= kMaxQuads || x1 <= x0 || y1 <= y0) return;
        static const float kFlat[4] = {1, 1, 1, 1};
        if (!grad) grad = kFlat;
        out[count++] = Quad{{x0, y0, x1, y1}, {grad[0], grad[1], grad[2], grad[3]}, {c[0], c[1], c[2], c[3] * alpha}, {1, radius, 0, 0}};
    }
    // A soft-edged box (shadow, glow): the box's edge blurred over `soft` pixels.
    void Soft(float x0, float y0, float x1, float y1, float radius, float soft, const float c[4]) {
        if (count >= kMaxQuads || x1 <= x0 || y1 <= y0) return;
        out[count++] = Quad{{x0 - soft, y0 - soft, x1 + soft, y1 + soft}, {1, 1, 1, 1}, {c[0], c[1], c[2], c[3] * alpha}, {1, radius, soft, soft}};
    }
    // Draws `s` with its top at y; returns the pen position after it.
    float Text(float x, float y, const std::wstring& s, FontId f, const float c[4]) {
        const Font& font = g.font[f];
        for (wchar_t ch : s) {
            const Glyph& gl = font.glyph[CharIndex(ch)];
            if (ch != L' ' && count < kMaxQuads) {
                const float gx = std::round(x) + float(gl.offset);
                out[count++] = Quad{{gx, y, gx + gl.w, y + gl.h}, {gl.u0, gl.v0, gl.u1, gl.v1}, {c[0], c[1], c[2], c[3] * alpha}, {0, 0, 0, 0}};
            }
            x += float(gl.advance);
        }
        return x;
    }
    float Width(const std::wstring& s, FontId f) const {
        float w = 0;
        for (wchar_t ch : s) w += float(g.font[f].glyph[CharIndex(ch)].advance);
        return w;
    }
};

struct View {
    int state = -1;  // 0 AMD, 1 ACTIVE, 2 STAND-IN, 3 comparing, 4 failed, -1 idle
    std::string upscaler, reason;
    uint32_t renderW = 0, renderH = 0, outW = 0, outH = 0;
    double upscaleMs = 0;  // 0: not measured in the last second
};

View ReadView() {
    View v;
    SharedStatus* s = Shared();
    if (!s) return v;
    const int64_t now = Now();
    AcquireSRWLockShared(&s->lock);
    auto recent = [&](int i) { return s->lastQpc[i] && Seconds(now - s->lastQpc[i]) < 1.0; };
    if (g_settings.hosted && s->lastState >= 0 && s->lastState < kStates && recent(s->lastState)) v.state = s->lastState;
    else if (s->lastState == int(wgfsr::UpscaleState::Failed) && recent(s->lastState)) v.state = 4;
    else if (s->compare && recent(int(wgfsr::UpscaleState::Compare))) v.state = 3;
    else if (recent(int(wgfsr::UpscaleState::Network))) v.state = 1;
    else if (recent(int(wgfsr::UpscaleState::StandIn))) v.state = 2;
    else if (recent(int(wgfsr::UpscaleState::Amd))) v.state = 0;
    v.upscaler = s->upscaler;
    v.reason = s->reason;
    v.renderW = s->renderW;
    v.renderH = s->renderH;
    v.outW = s->outW;
    v.outH = s->outH;
    if (v.state != 4 && s->upscaleQpc && Seconds(now - s->upscaleQpc) < 1.0) v.upscaleMs = s->upscaleMs;
    ReleaseSRWLockShared(&s->lock);
    return v;
}

// Top-left corner of a box of this size on the back buffer (setting overlay_position).
void Place(const Renderer& r, float boxW, float boxH, float* x0, float* y0) {
    const float margin = 14.0f * r.scale;
    *x0 = g_settings.right ? float(r.width) - margin - boxW : margin;
    *y0 = g_settings.bottom ? float(r.height) - 4.5f * margin - boxH : margin;  // bottom: clear of bottom HUD strips
    *x0 = std::max(0.0f, *x0);
    *y0 = std::max(0.0f, *y0);
}

// What the overlay shows besides the status: frame rate, frame time history, fade.
struct Frames {
    double fps = 0, ms = 0;     // on-screen numbers
    double upscaleMs = 0;       // on-screen upscaler time (0: none)
    float history[kHistory];    // seconds per Present, oldest first
    int count = 0;
    float alpha = 1;            // fade
    double time = 0;            // seconds, for the badge's pulse
};

// Upscaling factor and its usual preset name (the factors FSR and other upscalers share), if it is one of them.
std::wstring ScaleText(const View& v) {
    if (!v.renderW || !v.outW) return {};
    const double f = double(v.outW) / double(v.renderW);
    struct Preset {
        double factor;
        const wchar_t* name;
    } static const presets[] = {{1.0, L"Native AA"}, {1.3, L"Ultra Quality"}, {1.5, L"Quality"}, {1.7, L"Balanced"},
                                {2.0, L"Performance"}, {3.0, L"Ultra Performance"}};
    wchar_t buf[64];
    std::swprintf(buf, 64, L"%.2g\u00D7", f);
    for (const auto& p : presets)
        if (std::fabs(f - p.factor) < 0.03) return std::wstring(p.name) + L" \u00B7 " + buf;
    return buf;
}

// Fills `quads`; returns the quad count (0: nothing to draw) and the card's size. atOrigin: the card at (0, 0) (copy
// path), else at its place on the back buffer (overlay_position).
//
// Full panel: a rounded card with a soft shadow and a status-coloured accent bar; title "WrongGPU" with a status chip
// (ACTIVE / STAND-IN / OFF); label / value rows (upscaler, resolution with the scale factor, the upscaler's GPU time per
// frame, frame rate and frame time, a frame time graph; the reason when off); a hint line. Badge: a pill with a status
// dot (pulsing while the network runs), "WrongGPU", the upscaler, its time and the frame rate.
UINT Layout(Renderer& r, Quad* quads, const View& v, bool panel, const Frames& fr, bool atOrigin, UINT* boxWOut, UINT* boxHOut) {
    Painter p{r.glyphs, quads, 0, fr.alpha};
    const float s = r.scale;
    static const float kGreen[4] = {0.30f, 0.86f, 0.47f, 1}, kAmber[4] = {1.0f, 0.74f, 0.24f, 1}, kRed[4] = {1.0f, 0.36f, 0.36f, 1},
                       kGrey[4] = {0.55f, 0.56f, 0.60f, 1}, kWhite[4] = {0.95f, 0.96f, 0.97f, 1}, kLabel[4] = {0.62f, 0.64f, 0.69f, 1},
                       kCard[4] = {0.055f, 0.06f, 0.075f, 0.84f}, kEdge[4] = {1, 1, 1, 0.13f}, kShadow[4] = {0, 0, 0, 0.45f},
                       kCardGrad[4] = {1, 1, 1.45f, 0.85f}, kEdgeGrad[4] = {1.4f, 0.45f, 1, 1}, kBlue[4] = {0.40f, 0.66f, 1.0f, 1};
    const float* accent = v.state == 1 ? kGreen : v.state == 2 ? kAmber : v.state == 0 || v.state == 4 ? kRed : v.state == 3 ? kBlue : kGrey;
    const wchar_t* chip = v.state == 1 ? L"ACTIVE" : v.state == 2 ? L"STAND-IN" : v.state == 0 ? L"OFF" : v.state == 3 ? L"COMPARING" : v.state == 4 ? L"ERROR" : L"IDLE";
    std::wstring amdName(v.upscaler.begin(), v.upscaler.end());
    // AMD's public provider list may return only a version (e.g. "4.1.1").
    if (!amdName.empty() && amdName.front() >= L'0' && amdName.front() <= L'9') amdName = L"AMD FSR " + amdName;
    if (v.state < 0 && !panel) return 0;
    const bool network = v.upscaler.find("network") != std::string::npos;
    wchar_t buf[200];
    std::wstring fpsText, msText, upText, upShare;
    if (fr.fps > 0 && v.state >= 0) {
        std::swprintf(buf, 200, L"%.0f FPS", fr.fps);
        fpsText = buf;
        std::swprintf(buf, 200, L"%.1f ms", fr.ms);
        msText = buf;
    }
    if (fr.upscaleMs > 0 && v.state >= 0) {
        std::swprintf(buf, 200, L"%.2f ms", fr.upscaleMs);
        upText = buf;
        if (fr.ms > 0) {
            std::swprintf(buf, 200, L"%.0f %% of the frame", std::min(100.0, 100.0 * fr.upscaleMs / fr.ms));
            upShare = buf;
        }
    }
    const Font& body = r.glyphs.font[kBody];
    const Font& bold = r.glyphs.font[kBold];
    const Font& smallFont = r.glyphs.font[kSmall];
    // The wordmark, as in the logo: WRONG GPU with the red bar under GPU. Returns the pen position after it.
    static const float kLogoRed[4] = {0.992f, 0.275f, 0.235f, 1};
    const wchar_t* const kWordmark = L"WRONG GPU";
    auto wordmark = [&](float x, float y) {
        const float end = p.Text(x, y, kWordmark, kBold, kWhite);
        const float barX = x + p.Width(L"WRONG ", kBold), barY = y + float(bold.ascent) + 2 * s;
        p.Box(barX + 1 * s, barY, end - 1 * s, barY + std::max(2.0f, 2.5f * s), 0, kLogoRed);
        return end;
    };
    // The card: shadow (not in the copy path, whose texture is the card), hairline edge brighter at the top, body
    // lighter at the top.
    auto card = [&](float x0, float y0, float w, float h, float radius) {
        if (!atOrigin) p.Soft(x0, y0 + 3 * s, x0 + w, y0 + h + 3 * s, radius, 14 * s, kShadow);
        p.Box(x0 - 1, y0 - 1, x0 + w + 1, y0 + h + 1, radius + 1, kEdge, kEdgeGrad);
        p.Box(x0, y0, x0 + w, y0 + h, radius, kCard, kCardGrad);
    };

    if (!panel) {  // badge
        const float padX = 12 * s, h = float(bold.height) + 10 * s, dot = 8 * s;
#if WRONGGPU_MODEL_K
        std::wstring modelName = v.upscaler.find("Model K") != std::string::npos ? L"Model K" : L"Model M";
#else
        std::wstring modelName = L"Model M";
#endif
#if WRONGGPU_MODEL_L
        if (v.upscaler.find("Model L") != std::string::npos) modelName = L"Model L";
#endif
        std::wstring name = v.state == 1 ? (network ? modelName : L"active") : v.state == 2 ? L"stand-in"
                            : v.state == 3 ? amdName : v.state == 4 ? L"unavailable" : amdName.empty() ? L"off \u00B7 AMD FSR" : L"off \u00B7 " + amdName;
        const float wTitle = p.Width(kWordmark, kBold), wName = p.Width(name, kBody), wFps = fpsText.empty() ? 0 : p.Width(fpsText, kBody);
        const float gap = 8 * s;
        const float wUp = upText.empty() ? 0 : gap + p.Width(upText, kBody);
        const float w = padX + dot + gap + wTitle + gap + wName + wUp + (wFps > 0 ? gap * 3 + wFps : 0) + padX;
        float x0 = 0, y0 = 0;
        if (!atOrigin) {
            Place(r, w, h, &x0, &y0);
            y0 += (1 - fr.alpha) * 6 * s;  // slides in
        }
        card(x0, y0, w, h, h * 0.5f);
        float x = x0 + padX;
        const float dy = y0 + (h - dot) * 0.5f;
        if (v.state == 1) {  // live: a slow pulse around the dot
            const float pulse = 0.5f + 0.5f * float(std::sin(fr.time * 3.14159265 * 0.8));
            const float glow[4] = {accent[0], accent[1], accent[2], 0.25f + 0.35f * pulse};
            p.Soft(x - 1 * s, dy - 1 * s, x + dot + 1 * s, dy + dot + 1 * s, dot * 0.5f + 1 * s, 6 * s, glow);
        }
        p.Box(x, dy, x + dot, dy + dot, dot * 0.5f, accent);
        x += dot + gap;
        x = wordmark(x, y0 + (h - float(bold.height)) * 0.5f - 2) + gap;
        x = p.Text(x, y0 + (h - float(body.height)) * 0.5f - 1, name, kBody, accent);
        if (wUp > 0) x = p.Text(x + gap, y0 + (h - float(body.height)) * 0.5f - 1, upText, kBody, kWhite);
        if (wFps > 0) {
            const float divider[4] = {1, 1, 1, 0.16f};
            const float dx = x + gap * 1.5f;
            p.Box(dx - 0.5f * s, y0 + h * 0.28f, dx + 0.5f * s + 0.5f, y0 + h * 0.72f, 0, divider);
            p.Text(dx + gap * 1.5f, y0 + (h - float(body.height)) * 0.5f - 1, fpsText, kBody, kLabel);
        }
        *boxWOut = UINT(std::ceil(w));
        *boxHOut = UINT(std::ceil(h));
        return p.count;
    }

    // Full panel.
    struct Row {
        std::wstring label, value, suffix;  // suffix: in the label colour after the value
        bool graph = false;
    };
    std::vector<Row> rows;
    if (v.state >= 0) {
        rows.push_back({L"Upscaler", v.state == 0 ? (amdName.empty() ? std::wstring(L"AMD FSR") : amdName) + L" (WrongGPU off)"
                                     : v.state == 4 ? L"Unavailable (frame failed)" : v.state == 3 ? amdName
#if WRONGGPU_MODEL_L
                                     : network && v.upscaler.find("Model L") != std::string::npos ? L"Model L \u00B7 neural network"
#endif
#if WRONGGPU_MODEL_K
                                     : network ? (v.upscaler.find("Model K") != std::string::npos ? L"Model K \u00B7 neural network" : L"Model M \u00B7 neural network")
#else
                                     : network ? L"Model M \u00B7 neural network"
#endif
                                               : std::wstring(v.upscaler.begin(), v.upscaler.end()),
                        v.state == 3 ? L"comparison" : L""});
        if (v.renderW && v.outW) {
            std::swprintf(buf, 200, L"%u \u00D7 %u  \u2192  %u \u00D7 %u", v.renderW, v.renderH, v.outW, v.outH);
            rows.push_back({L"Resolution", buf, ScaleText(v)});
        }
        if (!upText.empty()) rows.push_back({L"Upscaler time", upText, upShare});
        if (!fpsText.empty()) rows.push_back({L"Frame rate", fpsText, msText});
        if (fr.count >= 16) rows.push_back({L"Frame time", {}, {}, true});
    } else {
        rows.push_back({L"Status", L"waiting for the game to use FSR upscaling", {}});
    }
    std::vector<std::wstring> reason;
    if ((v.state == 0 || v.state == 4) && !v.reason.empty()) {
        std::wstring text(v.reason.begin(), v.reason.end());
        const float maxW = 380 * s;
        while (!text.empty() && reason.size() < 4) {
            size_t cut = text.size();
            while (cut > 1 && p.Width(text.substr(0, cut), kSmall) > maxW) {
                const size_t sp = text.rfind(L' ', cut - 1);
                cut = sp == std::wstring::npos || sp == 0 ? cut - 1 : sp;
            }
            reason.push_back(text.substr(0, cut));
            text.erase(0, cut);
            while (!text.empty() && text[0] == L' ') text.erase(0, 1);
        }
    }
    std::wstring hint = g_settings.keyName + L" \u00B7 hide";
    if (g_settings.compareKey && v.state == 3) hint += L"     " + g_settings.compareName + L" \u00B7 back to WrongGPU";
    else if (g_settings.compareKey && (v.state == 1 || v.state == 2)) hint += L"     " + g_settings.compareName + L" \u00B7 compare with AMD FSR";

    const float pad = 14 * s, bar = 4 * s, rowGap = 3 * s, labelW = 104 * s, chipPadX = 8 * s, suffixGap = 12 * s;
    const float graphH = 30 * s;
    const float chipW = p.Width(chip, kSmall) + 2 * chipPadX, chipH = float(smallFont.height) + 4 * s;
    float valueW = 0;
    for (const auto& row : rows)
        if (!row.graph)
            valueW = std::max(valueW, p.Width(row.value, kBody) + (row.suffix.empty() ? 0 : suffixGap + p.Width(row.suffix, kSmall)));
    valueW = std::max(valueW, 220 * s);
    float contentW = p.Width(kWordmark, kBold) + 16 * s + chipW;
    contentW = std::max(contentW, labelW + valueW);
    for (const auto& line : reason) contentW = std::max(contentW, p.Width(line, kSmall));
    contentW = std::max(contentW, p.Width(hint, kSmall));
    valueW = contentW - labelW;
    const float titleH = float(bold.height) + 8 * s;
    float rowsH = 0;
    for (const auto& row : rows) rowsH += (row.graph ? graphH + 4 * s : float(body.height)) + rowGap;
    const float reasonH = reason.empty() ? 0 : 6 * s + float(reason.size()) * float(smallFont.height);
    const float hintH = 8 * s + float(smallFont.height);
    const float w = std::ceil(bar + pad + contentW + pad), h = std::ceil(pad + titleH + rowsH + reasonH + hintH + pad * 0.75f);
    float x0 = 0, y0 = 0;
    if (!atOrigin) {  // (the copy path's texture bounds the card)
        Place(r, w, h, &x0, &y0);
        y0 += (1 - fr.alpha) * 6 * s;
    }
    const float radius = 10 * s;
    card(x0, y0, w, h, radius);
    p.Box(x0 + 6 * s, y0 + pad, x0 + 6 * s + bar, y0 + h - pad, bar * 0.5f, accent);  // accent bar
    const float cx = x0 + bar + pad;
    float y = y0 + pad;
    wordmark(cx, y);
    const float chipX = cx + p.Width(kWordmark, kBold) + 16 * s, chipY = y + (float(bold.height) - chipH) * 0.5f + 1;
    const float chipFill[4] = {accent[0], accent[1], accent[2], 0.20f};
    p.Box(chipX, chipY, chipX + chipW, chipY + chipH, chipH * 0.5f, chipFill);
    p.Text(chipX + chipPadX, chipY + (chipH - float(smallFont.height)) * 0.5f, chip, kSmall, accent);
    y += titleH;
    for (const auto& row : rows) {
        const float labelY = y + float(body.ascent - smallFont.ascent);
        p.Text(cx, labelY, row.label, kSmall, kLabel);
        if (row.graph) {
            // Frame time graph: the newest frames on the right, one bar per frame; the scale puts the average at 40 %
            // of the height; frames taking over 1.5 x the average are drawn amber (stutter).
            const float gx0 = cx + labelW, gx1 = cx + labelW + valueW, gy0 = y + 2 * s, gy1 = y + 2 * s + graphH;
            const float well[4] = {1, 1, 1, 0.05f};
            p.Box(gx0, gy0, gx1, gy1, 4 * s, well);
            const float step = 3 * s, barW = std::max(1.0f, 2 * s);
            const int n = std::min(fr.count, int((gx1 - gx0 - 4 * s) / step));
            double sum = 0;
            for (int i = fr.count - n; i < fr.count; ++i) sum += fr.history[i];
            const float avg = float(sum / std::max(n, 1)), top = std::max(avg * 2.5f, 1e-4f);
            const float avgLine[4] = {1, 1, 1, 0.14f};
            const float ay = gy1 - 2 * s - (gy1 - gy0 - 4 * s) * (avg / top);
            p.Box(gx0 + 2 * s, ay, gx1 - 2 * s, ay + std::max(1.0f, s * 0.75f), 0, avgLine);
            const float barColour[4] = {accent[0], accent[1], accent[2], 0.85f};
            for (int k = 0; k < n; ++k) {
                const float t = fr.history[fr.count - n + k];
                const float bh = std::max(1.0f, (gy1 - gy0 - 4 * s) * std::min(1.0f, t / top));
                const float bx = gx1 - 2 * s - float(n - k) * step;
                p.Box(bx, gy1 - 2 * s - bh, bx + barW, gy1 - 2 * s, 0, t > avg * 1.5f ? kAmber : barColour);
            }
            y += graphH + 4 * s + rowGap;
            continue;
        }
        const float vx = p.Text(cx + labelW, y, row.value, kBody, kWhite);
        if (!row.suffix.empty()) p.Text(vx + suffixGap, labelY, row.suffix, kSmall, kLabel);
        y += float(body.height) + rowGap;
    }
    if (!reason.empty()) {
        y += 6 * s;
        for (const auto& line : reason) {
            p.Text(cx, y, line, kSmall, kRed);
            y += float(smallFont.height);
        }
    }
    y += 8 * s;
    p.Text(cx, y, hint, kSmall, kGrey);
    *boxWOut = UINT(w);
    *boxHOut = UINT(h);
    return p.count;
}

// ---------------------------------------------------------------------------------------------------------------
// Drawing in Present

Renderer* RendererFor(IDXGISwapChain3* sc) {
    Renderer* free = nullptr;
    Renderer* oldest = &g_state->renderers[0];
    for (auto& r : g_state->renderers) {
        if (r.key == sc) return &r;
        if (!r.key && !free) free = &r;
        if (r.lastUse < oldest->lastUse) oldest = &r;
    }
    Renderer* r = free ? free : oldest;
    r->Release();
    r->key = nullptr;
    return r;
}

// Records the text draw (quads already laid out) onto `target` (render target view `rtv`, size w x h) into list `l`.
void RecordText(Renderer& r, IDXGISwapChain3* sc, ID3D12GraphicsCommandList* l, D3D12_CPU_DESCRIPTOR_HANDLE rtv, UINT w, UINT h,
                ID3D12Resource* quads, UINT count) {
    l->SetGraphicsRootSignature(r.root);
    ID3D12DescriptorHeap* heaps[] = {r.srvHeap};
    l->SetDescriptorHeaps(1, heaps);
    struct {
        float invW, invH;
        uint32_t mode;
        float nits;
    } constants{1.0f / float(w), 1.0f / float(h), uint32_t(ColorModeFor(sc, r.format)), 203.0f};
    l->SetGraphicsRoot32BitConstants(0, 4, &constants, 0);
    l->SetGraphicsRootShaderResourceView(1, quads->GetGPUVirtualAddress());
    l->SetGraphicsRootDescriptorTable(2, r.srvHeap->GetGPUDescriptorHandleForHeapStart());
    D3D12_VIEWPORT vp{0, 0, float(w), float(h), 0, 1};
    D3D12_RECT sr{0, 0, LONG(w), LONG(h)};
    l->RSSetViewports(1, &vp);
    l->RSSetScissorRects(1, &sr);
    l->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    l->DrawInstanced(6, count, 0, 0);
}

// The font upload, once, in the first drawing list (DIRECT).
void UploadFont(Renderer& r, ID3D12GraphicsCommandList* l) {
    if (r.fontReady) return;
    const UINT pitch = (UINT(r.glyphs.texW) + 255u) & ~255u;
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = r.font;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = r.fontUpload;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint = {DXGI_FORMAT_R8_UNORM, UINT(r.glyphs.texW), UINT(r.glyphs.texH), 1, pitch};
    l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    const D3D12_RESOURCE_BARRIER b = Transition(r.font, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    l->ResourceBarrier(1, &b);
    r.fontReady = true;
}

// Direct path: the text drawn onto the back buffer on the present queue.
void DrawDirect(Renderer& r, IDXGISwapChain3* sc, const View& v, bool panel, const Frames& fr) {
    const UINT index = sc->GetCurrentBackBufferIndex();
    if (index >= r.buffers) return;
    auto& s = r.slot[index];
    if (r.fence->GetCompletedValue() < s.done) {  // the previous overlay on this buffer is still running: skip
        ++g_state->busy;
        return;
    }
    UINT boxW = 0, boxH = 0;
    const UINT quads = Layout(r, s.mapped, v, panel, fr, false, &boxW, &boxH);
    ID3D12Resource* back = nullptr;
    if (quads == 0 || FAILED(sc->GetBuffer(index, IID_PPV_ARGS(&back))) || !back) return;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = r.rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtv.ptr += SIZE_T(index) * r.rtvStep;
    r.device->CreateRenderTargetView(back, nullptr, rtv);
    s.alloc->Reset();
    ID3D12GraphicsCommandList* l = s.list;
    l->Reset(s.alloc, r.pso);
    UploadFont(r, l);
    D3D12_RESOURCE_BARRIER b = Transition(back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    l->ResourceBarrier(1, &b);
    RecordText(r, sc, l, rtv, r.width, r.height, s.quads, quads);
    b = Transition(back, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    l->ResourceBarrier(1, &b);
    if (SUCCEEDED(l->Close())) {
        ID3D12CommandList* lists[] = {l};
        r.queue->ExecuteCommandLists(1, lists);
        r.queue->Signal(r.fence, ++r.fenceValue);
        s.done = r.fenceValue;
        ++g_state->drawn;
    }
    back->Release();
}

// Copy path: render the panel into a free texture on the overlay's own queue; copy the newest finished one onto the
// back buffer on the present queue. The present queue never waits for another queue.
void DrawCopy(Renderer& r, IDXGISwapChain3* sc, const View& v, bool panel, const Frames& fr) {
    const UINT64 ownDone = r.ownFence->GetCompletedValue(), copyDone = r.fence->GetCompletedValue();
    // 1. Render into the free panel texture rendered longest ago.
    Renderer::Panel* target = nullptr;
    for (auto& p : r.panel)
        if (ownDone >= p.rendered && copyDone >= p.copied && (!target || p.seq < target->seq)) target = &p;
    if (target) {
        UINT boxW = 0, boxH = 0;
        const UINT quads = Layout(r, target->mapped, v, panel, fr, true, &boxW, &boxH);
        if (quads == 0) return;
        const UINT i = UINT(target - r.panel);
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = r.rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rtv.ptr += SIZE_T(i) * r.rtvStep;
        target->alloc->Reset();
        ID3D12GraphicsCommandList* l = target->list;
        l->Reset(target->alloc, r.pso);
        UploadFont(r, l);
        D3D12_RESOURCE_BARRIER b = Transition(target->tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        l->ResourceBarrier(1, &b);
        const float clear[4] = {0, 0, 0, 1};
        D3D12_RECT box{0, 0, LONG(boxW), LONG(boxH)};
        l->ClearRenderTargetView(rtv, clear, 1, &box);
        RecordText(r, sc, l, rtv, r.panelW, r.panelH, target->quads, quads);
        b = Transition(target->tex, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COMMON);
        l->ResourceBarrier(1, &b);
        if (SUCCEEDED(l->Close())) {
            ID3D12CommandList* lists[] = {l};
            r.ownQueue->ExecuteCommandLists(1, lists);
            r.ownQueue->Signal(r.ownFence, ++r.ownValue);
            target->rendered = r.ownValue;
            target->seq = ++r.panelSeq;
            target->boxW = boxW;
            target->boxH = boxH;
        }
    } else if (!panel && v.state < 0) {
        return;
    }
    // 2. Copy the newest panel the GPU has finished (per the CPU's view of the fence) onto the back buffer.
    Renderer::Panel* ready = nullptr;
    for (auto& p : r.panel)
        if (p.seq && ownDone >= p.rendered && (!ready || p.seq > ready->seq)) ready = &p;
    if (!ready || ready->boxW == 0) return;
    const UINT index = sc->GetCurrentBackBufferIndex();
    if (index >= r.buffers) return;
    auto& s = r.slot[index];
    if (copyDone < s.done) {
        ++g_state->busy;
        return;
    }
    ID3D12Resource* back = nullptr;
    if (FAILED(sc->GetBuffer(index, IID_PPV_ARGS(&back))) || !back) return;
    float x0 = 0, y0 = 0;
    Place(r, float(ready->boxW), float(ready->boxH), &x0, &y0);
    const UINT w = std::min(ready->boxW, r.width - std::min(r.width, UINT(x0)));
    const UINT h = std::min(ready->boxH, r.height - std::min(r.height, UINT(y0)));
    s.alloc->Reset();
    ID3D12GraphicsCommandList* l = s.list;
    l->Reset(s.alloc, nullptr);
    D3D12_RESOURCE_BARRIER in[2] = {Transition(back, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST),
                                    Transition(ready->tex, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    l->ResourceBarrier(2, in);
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = back;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = ready->tex;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_BOX srcBox{0, 0, 0, w, h, 1};
    if (w && h) l->CopyTextureRegion(&dst, UINT(x0), UINT(y0), 0, &src, &srcBox);
    D3D12_RESOURCE_BARRIER outB[2] = {Transition(back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                                      Transition(ready->tex, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)};
    l->ResourceBarrier(2, outB);
    if (SUCCEEDED(l->Close())) {
        ID3D12CommandList* lists[] = {l};
        r.queue->ExecuteCommandLists(1, lists);
        r.queue->Signal(r.fence, ++r.fenceValue);
        s.done = r.fenceValue;
        ready->copied = r.fenceValue;
        ++g_state->drawn;
    }
    back->Release();
}

void Draw(IDXGISwapChain* swapChain) {
    if (g_state->disabled || g_settings.mode == 0) return;
    std::lock_guard<std::mutex> lock(g_state->mutex);

    // Frame rate from Present intervals; toggle key (only while the game's window has the focus).
    const int64_t now = Now();
    if (g_state->lastPresent) {
        const double dt = Seconds(now - g_state->lastPresent);
        if (dt > 0 && dt < 1.0) {
            g_state->frameTime = g_state->frameTime > 0 ? g_state->frameTime * 0.95 + dt * 0.05 : dt;
            g_state->history[g_state->historyHead] = float(dt);
            g_state->historyHead = (g_state->historyHead + 1) % kHistory;
            g_state->historyCount = std::min(g_state->historyCount + 1, kHistory);
        }
    }
    g_state->lastPresent = now;
    const bool refresh = now >= g_state->textNext && g_state->frameTime > 0;  // steady numbers: refreshed 4 times a second
    if (refresh) {
        g_state->shownFps = 1.0 / g_state->frameTime;
        g_state->shownMs = g_state->frameTime * 1000.0;
        g_state->textNext = now + int64_t(0.25 / Seconds(1));
    }
    if (!g_state->shownAt) g_state->shownAt = now;
    DWORD pid = 0;
    GetWindowThreadProcessId(GetForegroundWindow(), &pid);
    const bool down = pid == GetCurrentProcessId() && (GetAsyncKeyState(g_settings.key) & 0x8000) != 0;
    if (down && !g_state->keyDown) Toggle(now);
    g_state->keyDown = down;
    const bool compareDown = g_settings.compareKey && pid == GetCurrentProcessId() && (GetAsyncKeyState(g_settings.compareKey) & 0x8000) != 0;
    if (compareDown && !g_state->compareKeyDown) {
        if (SharedStatus* shared = Shared()) {
            const LONG on = shared->compare ? 0 : 1;
            InterlockedExchange(&shared->compare, on);
            Log::Get().Write("overlay: %s (upscaler time until now %.3f ms, %.1f FPS)",
                             on ? "comparison: AMD's FSR upscales" : "comparison over: WrongGPU upscales", g_state->shownUpscaleMs,
                             g_state->shownFps);
            if (!g_state->visible) {  // the switch shows itself
                g_state->visible = true;
                g_state->shownAt = now;
            }
        }
    }
    g_state->compareKeyDown = compareDown;
    float alpha = 1;
    if (g_state->visible) {
        alpha = float(std::min(1.0, Seconds(now - g_state->shownAt) / kFade));
    } else {
        const double t = Seconds(now - g_state->hiddenAt);
        if (t >= kFade) return;
        alpha = float(1.0 - t / kFade);
    }

    const View v = ReadView();
    if (refresh || v.upscaleMs <= 0) g_state->shownUpscaleMs = v.upscaleMs;
    if (v.state != g_state->lastShownState) {  // state changed: the full panel again for 10 s
        Log::Get().Write("overlay: state %d -> %d", g_state->lastShownState, v.state);
        g_state->lastShownState = v.state;
        if (v.state >= 0) {
            if (g_settings.mode == 1 && now >= g_state->panelUntil + int64_t(kFade / Seconds(1))) g_state->shownAt = now;  // badge -> panel
            g_state->panelUntil = now + int64_t(10.0 / Seconds(1));
        }
    }
    bool panel = g_settings.mode == 1 && (v.state < 0 || now < g_state->panelUntil);
    if (g_settings.mode == 1 && v.state >= 0 && !panel) {  // panel fades out, then the badge fades in
        const double t = Seconds(now - g_state->panelUntil);
        if (t < kFade) {
            panel = true;
            alpha *= float(1.0 - t / kFade);
        } else if (t < 2 * kFade) {
            alpha *= float((t - kFade) / kFade);
        }
    }
    if (v.state < 0 && !panel) return;

    IDXGISwapChain3* sc = nullptr;
    if (FAILED(swapChain->QueryInterface(IID_PPV_ARGS(&sc))) || !sc) {
        LogOnce("swap chain without IDXGISwapChain3 not drawn");
        return;
    }
    struct ScRef {
        IDXGISwapChain3* p;
        ~ScRef() { p->Release(); }
    } scRef{sc};
    ID3D12Resource* probe = nullptr;  // only D3D12 swap chains have ID3D12Resource buffers
    if (FAILED(sc->GetBuffer(0, IID_PPV_ARGS(&probe))) || !probe) {
        LogOnce("swap chain without D3D12 buffers not drawn");
        return;
    }
    probe->Release();
    ID3D12CommandQueue* presentQueue = SwapChainQueueOf(swapChain);
    if (!presentQueue) return;  // learnt during this Present; drawn from the next one
    struct QueueRef {
        ID3D12CommandQueue* p;
        ~QueueRef() { p->Release(); }
    } queueRef{presentQueue};
    if (++g_state->presents % 3600 == 1)
        Log::Get().Write("overlay: %llu presents, %llu drawn, %llu skipped (busy)", static_cast<unsigned long long>(g_state->presents),
                         static_cast<unsigned long long>(g_state->drawn), static_cast<unsigned long long>(g_state->busy));
    DXGI_SWAP_CHAIN_DESC1 desc{};
    DXGI_SWAP_CHAIN_DESC old{};
    sc->GetDesc1(&desc);
    sc->GetDesc(&old);
    Renderer* r = RendererFor(sc);
    const bool queueChanged = r->key == sc && r->queue && r->queue != presentQueue;
    if (r->key != sc || r->width != desc.Width || r->height != desc.Height || r->format != desc.Format || r->buffers != old.BufferCount ||
        queueChanged) {
        if (!Create(*r, sc, presentQueue, desc, old.BufferCount)) {
            r->Release();
            r->key = sc;  // remembered as unsupported until it changes
            r->width = desc.Width;
            r->height = desc.Height;
            r->format = desc.Format;
            r->buffers = old.BufferCount;
        }
    }
    if (!r->pso) return;
    r->lastUse = now;
    Frames fr;
    fr.fps = g_state->shownFps;
    fr.ms = g_state->shownMs;
    fr.upscaleMs = g_state->shownUpscaleMs;
    fr.count = g_state->historyCount;
    for (int i = 0; i < fr.count; ++i)
        fr.history[i] = g_state->history[(g_state->historyHead - fr.count + i + kHistory) % kHistory];
    fr.time = Seconds(now);
    fr.alpha = alpha;
    if (r->copyPath) {  // the panel texture is opaque: no fades
        if (!g_state->visible) return;
        fr.alpha = 1;
        DrawCopy(*r, sc, v, panel, fr);
    } else {
        DrawDirect(*r, sc, v, panel, fr);
    }
}

// Structured exception guard: an access violation or other exception in the overlay disables it instead of crashing.
#if defined(_MSC_VER)
void DrawGuarded(IDXGISwapChain* sc) {
    __try {
        Draw(sc);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        g_state->disabled = true;
        Log::Get().Write("overlay: exception 0x%08lx while drawing; overlay disabled for this session", GetExceptionCode());
    }
}
#else
void DrawGuarded(IDXGISwapChain* sc) {
    try {
        Draw(sc);
    } catch (...) {
        g_state->disabled = true;
        Log::Get().Write("overlay: exception while drawing; overlay disabled for this session");
    }
}
#endif

void ForgetSwapChain(IUnknown* swapChain) {
    IDXGISwapChain3* sc = nullptr;
    if (FAILED(swapChain->QueryInterface(IID_PPV_ARGS(&sc))) || !sc) return;
    {
        std::lock_guard<std::mutex> lock(g_state->mutex);
        for (auto& r : g_state->renderers)
            if (r.key == sc) {
                r.Release();
                r.key = nullptr;
                r.width = r.height = r.buffers = 0;
            }
    }
    sc->Release();
}

HRESULT STDMETHODCALLTYPE HPresent(IDXGISwapChain* sc, UINT sync, UINT flags) {
    if (!(flags & DXGI_PRESENT_TEST)) DrawGuarded(sc);
    t_inPresent = sc;
    const HRESULT hr = reinterpret_cast<PfnPresent>(Original(sc, kSlotPresent))(sc, sync, flags);
    t_inPresent = nullptr;
    return hr;
}

HRESULT STDMETHODCALLTYPE HPresent1(IDXGISwapChain1* sc, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS* params) {
    if (!(flags & DXGI_PRESENT_TEST)) DrawGuarded(sc);
    t_inPresent = sc;
    const HRESULT hr = reinterpret_cast<PfnPresent1>(Original(sc, kSlotPresent1))(sc, sync, flags, params);
    t_inPresent = nullptr;
    return hr;
}

HRESULT STDMETHODCALLTYPE HResizeBuffers(IDXGISwapChain* sc, UINT count, UINT w, UINT h, DXGI_FORMAT f, UINT flags) {
    ForgetSwapChain(sc);
    return reinterpret_cast<PfnResizeBuffers>(Original(sc, kSlotResizeBuffers))(sc, count, w, h, f, flags);
}

HRESULT STDMETHODCALLTYPE HResizeBuffers1(IDXGISwapChain3* sc, UINT count, UINT w, UINT h, DXGI_FORMAT f, UINT flags,
                                          const UINT* mask, IUnknown* const* queues) {
    ForgetSwapChain(sc);
    return reinterpret_cast<PfnResizeBuffers1>(Original(sc, kSlotResizeBuffers1))(sc, count, w, h, f, flags, mask, queues);
}

HRESULT STDMETHODCALLTYPE HSetColorSpace1(IDXGISwapChain3* sc, DXGI_COLOR_SPACE_TYPE space) {
    const HRESULT hr = reinterpret_cast<PfnSetColorSpace1>(Original(sc, kSlotSetColorSpace1))(sc, space);
    if (SUCCEEDED(hr)) {
        std::lock_guard<std::mutex> lock(g_state->mutex);
        int slot = -1;
        for (int i = 0; i < 4; ++i)
            if (g_state->colorSpaceKey[i] == sc || (slot < 0 && !g_state->colorSpaceKey[i])) slot = i;
        if (slot < 0) slot = 0;
        g_state->colorSpaceKey[slot] = sc;
        g_state->colorSpace[slot] = space;
    }
    return hr;
}

struct SlotHook {
    int slot;
    void* hook;
};

bool PatchVtbl(Family& family, void** vt, const std::vector<SlotHook>& hooks) {
    const int n = family.count.load();
    for (int i = 0; i < n; ++i)
        if (family.entries[i].vtbl == vt) return true;
    if (n >= 4) return false;
    PatchedVtbl& e = family.entries[n];
    e.vtbl = vt;
    for (const auto& h : hooks) e.orig[h.slot] = vt[h.slot];
    family.count.store(n + 1, std::memory_order_release);
    for (const auto& h : hooks) {
        DWORD old = 0, unused = 0;
        if (!VirtualProtect(&vt[h.slot], sizeof(void*), PAGE_READWRITE, &old)) {
            Log::Get().Write("overlay: VirtualProtect of vtable entry %d failed (error %lu)", h.slot, GetLastError());
            continue;
        }
        if (e.orig[h.slot] != vt[h.slot]) e.orig[h.slot] = vt[h.slot];
        InterlockedExchangePointer(&vt[h.slot], h.hook);
        VirtualProtect(&vt[h.slot], sizeof(void*), old, &unused);
    }
    return true;
}

// The swap chain vtable comes from a 2-buffer swap chain on a hidden window, created on the game's device and released
// again (every DXGI flip-model swap chain of the process shares it).
bool InstallHooks(ID3D12Device* device) {
    using PfnCreateFactory = HRESULT(WINAPI*)(UINT, REFIID, void**);
    HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
    auto createFactory = dxgi ? ProcAddress<PfnCreateFactory>(dxgi, "CreateDXGIFactory2") : nullptr;
    IDXGIFactory2* factory = nullptr;
    if (!createFactory || FAILED(createFactory(0, IID_PPV_ARGS(&factory)))) {
        Log::Get().Write("overlay: CreateDXGIFactory2 failed; overlay off");
        return false;
    }
    ID3D12CommandQueue* queue = nullptr;
    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    const HRESULT hq = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"WrongGPUOverlayProbe";
    RegisterClassExW(&wc);
    HWND wnd = CreateWindowExW(0, wc.lpszClassName, L"", WS_OVERLAPPED, 0, 0, 16, 16, nullptr, nullptr, wc.hInstance, nullptr);
    bool ok = false;
    if (SUCCEEDED(hq) && wnd) {
        DXGI_SWAP_CHAIN_DESC1 d{};
        d.Width = 16;
        d.Height = 16;
        d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        d.SampleDesc.Count = 1;
        d.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        d.BufferCount = 2;
        d.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        IDXGISwapChain1* sc1 = nullptr;
        if (SUCCEEDED(factory->CreateSwapChainForHwnd(queue, wnd, &d, nullptr, nullptr, &sc1)) && sc1) {
            IDXGISwapChain3* sc3 = nullptr;
            if (SUCCEEDED(sc1->QueryInterface(IID_PPV_ARGS(&sc3))) && sc3) {
                ok = PatchVtbl(g_swapChainVt, *reinterpret_cast<void***>(sc3),
                               {{kSlotPresent, reinterpret_cast<void*>(&HPresent)},
                                {kSlotResizeBuffers, reinterpret_cast<void*>(&HResizeBuffers)},
                                {kSlotPresent1, reinterpret_cast<void*>(&HPresent1)},
                                {kSlotSetColorSpace1, reinterpret_cast<void*>(&HSetColorSpace1)},
                                {kSlotResizeBuffers1, reinterpret_cast<void*>(&HResizeBuffers1)}}) &&
                     PatchVtbl(g_queueVt, *reinterpret_cast<void***>(queue),
                               {{kSlotExecuteCommandLists, reinterpret_cast<void*>(&HExecuteCommandLists)},
                                {kSlotSignal, reinterpret_cast<void*>(&HSignal)},
                                {kSlotWait, reinterpret_cast<void*>(&HWait)}});
                sc3->Release();
            }
            sc1->Release();
        }
    }
    if (wnd) DestroyWindow(wnd);
    SafeRelease(queue);
    factory->Release();
    Log::Get().Write(ok ? "overlay: hooks installed (swap chain Present, Present1, ResizeBuffers, ResizeBuffers1, SetColorSpace1; queue ExecuteCommandLists)"
                        : "overlay: could not hook the swap chain; overlay off");
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------
// Upscaler time: two GPU timestamps per upscale dispatch, recorded into the game's command list before and after the
// dispatch (with WrongGPU's network that span holds the input copies, the hand-over to the network and back, and the
// output copy); resolved into a readback buffer that is read kTimerSlots / 2 dispatches later, without waiting.

struct GpuTimer {
    SRWLOCK lock = SRWLOCK_INIT;
    ID3D12Device* device = nullptr;  // referenced: the heap below belongs to it
    ID3D12QueryHeap* heap = nullptr;
    ID3D12Resource* readback = nullptr;
    const uint64_t* results = nullptr;  // mapped readback: begin, end per slot
    double ticksPerMs = 0;
    uint64_t next = 0;
    uint64_t lastBegin[kTimerSlots] = {};
    bool failed = false;  // set up failed for this device: no timing

    void Release() {
        if (readback && results) readback->Unmap(0, nullptr);
        results = nullptr;
        SafeRelease(readback);
        SafeRelease(heap);
        SafeRelease(device);
    }
    bool Setup(ID3D12Device* d) {
        Release();
        failed = true;
        device = d;
        d->AddRef();
        std::fill(std::begin(lastBegin), std::end(lastBegin), 0);
        // The timestamp frequency belongs to a queue; a short-lived DIRECT queue of the device tells it (the game's
        // DIRECT and COMPUTE queues count at the same rate on the GPUs we run on).
        D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT};
        ID3D12CommandQueue* q = nullptr;
        UINT64 freq = 0;
        if (FAILED(d->CreateCommandQueue(&qd, IID_PPV_ARGS(&q))) || !q) return false;
        const HRESULT hrFreq = q->GetTimestampFrequency(&freq);
        q->Release();
        if (FAILED(hrFreq) || freq == 0) return false;
        ticksPerMs = double(freq) / 1000.0;
        D3D12_QUERY_HEAP_DESC hd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, kTimerSlots * 2, 0};
        if (FAILED(d->CreateQueryHeap(&hd, IID_PPV_ARGS(&heap))) || !heap) return false;
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = kTimerSlots * 2 * sizeof(uint64_t);
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                              IID_PPV_ARGS(&readback))) || !readback)
            return false;
        void* mapped = nullptr;
        const D3D12_RANGE all{0, SIZE_T(rd.Width)};
        if (FAILED(readback->Map(0, &all, &mapped)) || !mapped) return false;
        results = static_cast<const uint64_t*>(mapped);
        failed = false;
        return true;
    }
};
GpuTimer g_timer;
thread_local int t_timerDepth = 0;
thread_local UINT t_timerSlot = UINT(-1);  // this thread's open measurement
thread_local bool t_timerOwner = false;    // this DLL's outermost call holds SharedStatus::timerThread

void PublishUpscalerMs(double ms) {
    SharedStatus* s = Shared();
    if (!s) return;
    const int64_t now = Now();
    AcquireSRWLockExclusive(&s->lock);
    if (s->upscaleSkip > 0) {
        --s->upscaleSkip;
    } else {
        const bool recent = !s->upscaleFresh && s->upscaleQpc && Seconds(now - s->upscaleQpc) < 1.0;
        s->upscaleMs = recent ? s->upscaleMs * 0.9 + ms * 0.1 : ms;
        s->upscaleQpc = now;
        s->upscaleFresh = false;
    }
    ReleaseSRWLockExclusive(&s->lock);
}

void GpuTimerImpl(void* commandList, bool end) {
    auto* l = static_cast<ID3D12GraphicsCommandList*>(commandList);
    if (!end) {
        if (++t_timerDepth != 1) return;
        t_timerSlot = UINT(-1);
        t_timerOwner = false;
        SharedStatus* s = Shared();
        if (g_settings.mode == 0 || !l || !s) return;
        // FSR 3.0's DLL calls its upscaler DLL (both WrongGPU's): only the outermost dispatch of the thread is timed.
        if (InterlockedCompareExchange(&s->timerThread, LONG(GetCurrentThreadId()), 0) != 0) return;
        t_timerOwner = true;
        const D3D12_COMMAND_LIST_TYPE type = l->GetType();
        if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) return;
        ID3D12Device* d = nullptr;
        if (FAILED(l->GetDevice(IID_PPV_ARGS(&d))) || !d) return;
        AcquireSRWLockExclusive(&g_timer.lock);
        if (g_timer.device != d && !g_timer.Setup(d)) Log::Get().Write("overlay: GPU timestamps unavailable; no upscaler time");
        if (!g_timer.failed) {
            t_timerSlot = UINT(g_timer.next++ % kTimerSlots);
            l->EndQuery(g_timer.heap, D3D12_QUERY_TYPE_TIMESTAMP, t_timerSlot * 2);
        }
        ReleaseSRWLockExclusive(&g_timer.lock);
        d->Release();
        return;
    }
    if (t_timerDepth > 0 && --t_timerDepth != 0) return;
    const UINT slot = t_timerSlot;
    t_timerSlot = UINT(-1);
    if (t_timerOwner) {
        t_timerOwner = false;
        if (SharedStatus* s = Shared()) InterlockedExchange(&s->timerThread, 0);
    }
    if (slot == UINT(-1)) return;
    double ms = -1;
    AcquireSRWLockExclusive(&g_timer.lock);
    if (!g_timer.failed) {
        l->EndQuery(g_timer.heap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 2 + 1);
        l->ResolveQueryData(g_timer.heap, D3D12_QUERY_TYPE_TIMESTAMP, slot * 2, 2, g_timer.readback, UINT64(slot) * 2 * sizeof(uint64_t));
        // The slot written kTimerSlots / 2 dispatches ago. A slot the GPU has not resolved again since the last read
        // keeps its begin value (skipped); a half-written one has end < begin or an absurd span (skipped).
        const UINT old = (slot + kTimerSlots / 2) % kTimerSlots;
        const uint64_t b = g_timer.results[old * 2], e = g_timer.results[old * 2 + 1];
        if (b && e > b && b != g_timer.lastBegin[old]) {
            g_timer.lastBegin[old] = b;
            ms = double(e - b) / g_timer.ticksPerMs;
        }
    }
    ReleaseSRWLockExclusive(&g_timer.lock);
    if (ms >= 0 && ms < 100) PublishUpscalerMs(ms);
}

}  // namespace

void OverlayGpuTimer(void* commandList, bool end) {
#if defined(_MSC_VER)
    __try {
        GpuTimerImpl(commandList, end);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        t_timerDepth = 0;
        t_timerSlot = UINT(-1);
        if (t_timerOwner) {
            t_timerOwner = false;
            if (SharedStatus* s = Shared()) InterlockedExchange(&s->timerThread, 0);
        }
    }
#else
    GpuTimerImpl(commandList, end);
#endif
}

bool OverlayCompareAmd() {
    SharedStatus* s = Shared();
    return s && s->compare != 0;
}

void OverlayToggle() {
    if (g_state) Toggle(Now());
}

double OverlayUpscalerMs() {
    SharedStatus* s = Shared();
    if (!s) return 0;
    AcquireSRWLockShared(&s->lock);
    const double ms = s->upscaleQpc && Seconds(Now() - s->upscaleQpc) < 1.0 ? s->upscaleMs : 0;
    ReleaseSRWLockShared(&s->lock);
    return ms;
}

void ConfigureOverlay(const wgfsr::Config& config) {
    g_settings.hosted = config.hosted;
    g_settings.mode = config.overlay == "off" ? 0 : config.overlay == "badge" ? 2 : 1;
    const int key = wgfsr::OverlayKeyCode(config.overlayKey);
    g_settings.key = key ? key : 0x2D;
    g_settings.keyName = KeyLabel(key ? config.overlayKey : "insert");
    g_settings.compareKey = config.compareKey == "off" ? 0 : wgfsr::OverlayKeyCode(config.compareKey);
    g_settings.compareName = KeyLabel(config.compareKey);
    g_settings.right = config.overlayPosition.find("right") != std::string::npos;
    g_settings.bottom = config.overlayPosition.find("bottom") != std::string::npos;
}

bool ReadOverlayProviderStatus(wgfsr::UpscaleStatus& status) {
    const View v = ReadView();
    if (v.state < 0) return false;
    status.state = wgfsr::UpscaleState(v.state);
    status.upscaler = v.upscaler;
    status.reason = v.reason;
    status.renderWidth = v.renderW;
    status.renderHeight = v.renderH;
    status.outputWidth = v.outW;
    status.outputHeight = v.outH;
    status.device = nullptr; // this read-only presentation seam never uses the device
    return true;
}

void OverlayStatus(const wgfsr::UpscaleStatus& status) {
    SharedStatus* s = Shared();
    if (!s) return;
    const int i = int(status.state);
    if (i < 0 || i >= kStates) return;
    AcquireSRWLockExclusive(&s->lock);
    s->lastQpc[i] = Now();
    ++s->frames[i];
    if (s->lastState != i) {  // another upscaler: its time is measured afresh
        s->lastState = i;
        s->upscaleSkip = int(kTimerSlots / 2) + 2;
        s->upscaleFresh = true;
    }
    if (status.renderWidth) {
        s->renderW = status.renderWidth;
        s->renderH = status.renderHeight;
        s->outW = status.outputWidth;
        s->outH = status.outputHeight;
    }
    CopyText(s->upscaler, sizeof s->upscaler, status.upscaler);  // (Compare: AMD's name)
    if (status.state == wgfsr::UpscaleState::Amd || status.state == wgfsr::UpscaleState::Failed)
        CopyText(s->reason, sizeof s->reason, status.reason);
    ReleaseSRWLockExclusive(&s->lock);

    // The first DLL with a device installs the hooks (once per process).
    if (g_settings.mode == 0 || !status.device || s->owner) return;
    if (InterlockedCompareExchange(&s->owner, 1, 0) != 0) return;
    HMODULE self = nullptr;  // the hooks point into this DLL: keep it loaded
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN, reinterpret_cast<LPCWSTR>(&OverlayStatus),
                       &self);
    g_state = new State();
    if (!InstallHooks(static_cast<ID3D12Device*>(status.device))) g_state->disabled = true;
}

}  // namespace wgwin
