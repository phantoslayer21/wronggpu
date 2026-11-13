// SPDX-License-Identifier: MIT
// End-to-end test of the built amd_fidelityfx_dx12.dll through its exports, with a real D3D12 device (Windows, or
// Wine with vkd3d and a Vulkan driver such as lavapipe). AMD's DLL is replaced by the test double
// amd_fidelityfx_dx12.wronggpu-original.dll (tests/fake_original.cpp).
//
//   test_dll_d3d12 <mode>    run from the folder holding the three files; exit 0 pass, 1 fail, 77 no D3D12 device
//
// Modes (one process per mode, because the DLLs read wronggpu.ini once):
//   upscale     stand-in bilinear upscale with jitter compensation vs a CPU reference and vs the true scene
//   lanczos     the same with the Lanczos filter
//   warpdiff    two frames of a moving scene; the warp-difference debug view must be near zero (FORMATS.md motion
//               convention), so a wrong motion conversion would show
//   motionsign  the same with motion_sign = -1 in the .ini: the difference must be large (the check can fail)
//   forward     no GPU work: frame generation, global queries and an upscaler create without a D3D12 backend go to
//               the fake AMD DLL
//   amd         upscaler = amd: the upscaler context is created, dispatched and destroyed by the fake AMD DLL
//   view_motion, view_depth, view_jitter   the debug views write the expected values
//   srgb        an sRGB output (typeless texture, FidelityFX format R8G8B8A8_SRGB) is written through a UNORM view
// FSR 3.0 component DLLs (WrongGPU's ffx_fsr3upscaler_x64.dll and ffx_fsr3_x64.dll over fake AMD originals):
//   fsr30_log              upscaler = amd: every export is forwarded and logged with its caller
//   fsr30_upscaler         the game calls the upscaler DLL directly, no frame generation: stand-in only
//   fsr30_upscaler_shared  the same with frame-generation resources: AMD's upscaler first, then the stand-in
//   fsr30_fsr3_inner       the game calls ffx_fsr3, which calls the upscaler DLL: taken over in the upscaler DLL
//   fsr30_fsr3_static      ffx_fsr3 does not call the upscaler DLL: taken over after ffxFsr3ContextDispatchUpscale
//   fsr30_fsr3_upscaling_only  as fsr30_fsr3_inner with an UPSCALING_ONLY context (Cyberpunk 2077 without frame
//                          generation): AMD's upscaler does not run at all
// Same-frame split (interface version 2; WrongGPU's D3D12 hooks):
//   hooks_trace    upscaler = amd, d3d12_hooks = on: the submission trace of the FSR list (test 20-T4 offline)
//   split_standin  stand-in with split_test = 1 (test 20-T5 offline): the first half runs after a list the game
//                  submits ahead of it, the game's graphics state survives, the upscale is right; two frames
//   split_fsr30    the same through the FSR 3.0 upscaler DLL: frame 0 is not ready (no split), frame 1 is split
//   split_release  a split list reset without submission, and one submitted after split_timeout_ms: released from
//                  the CPU, nothing hangs
//   split_control  the split_standin scenario without split_test (the scenario's own expectations, no hooks)
// upscaler = network with the test double of the runtime (tests/fake_runtime.cpp, wronggpu_runtime.dll):
//   network_missing        runtime_dll not in the folder: said so, AMD upscales
//   network_create_fail    the runtime's create fails: its own error line and the result in words, AMD upscales
//   network_dispatch_fail  the runtime fails on frame 1: in words; AMD takes the context from then on
//   network_split          a version 2 runtime with WG_BACKEND_NEEDS_SPLIT splits every frame
//   network_v1             a runtime that offers only interface version 1 gets version 1 structures
//   network_bad            a runtime whose tables lack functions is refused, with the reason; AMD upscales
//   network_cyberpunk_fail Cyberpunk's FSR 3.0 path: the runtime fails on frame 1, AMD's upscaler takes over
// By hand, with the real runtime (not ctests): park_real (a game's dynamic resolution), switch_real (a player
// switching upscalers in an upscaler mod's menu; GPU memory after every step; see RunSwitchReal).
// With MSVC, the D3D12 debug layer is on when installed (WG_TEST_D3D12_DEBUG=0 turns it off); its errors fail the test.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3dcommon.h>
#include <dxgi1_4.h>
#if defined(_MSC_VER)
#include <d3d12sdklayers.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ffx_api/dx12/ffx_api_dx12.h"
#include "ffx_api/ffx_api.h"
#include "ffx_api/ffx_framegeneration.h"
#include "ffx_api/ffx_upscale.h"
#include "../src/fsr30_types.h"

namespace {

int g_failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            ++g_failures;                                               \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);         \
        }                                                               \
    } while (0)

std::wstring g_dir;
std::wstring g_dllName = L"amd_fidelityfx_dx12.dll";  // switch_real: wronggpu_ffx_dx12.dll (hosted by an upscaler mod)

std::wstring ExeDirectory() {
    wchar_t path[MAX_PATH]{};
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring p(path);
    return p.substr(0, p.find_last_of(L"\\/"));
}

void WriteFileText(const std::wstring& path, const std::string& text) {
    HANDLE f = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return;
    DWORD n = 0;
    WriteFile(f, text.data(), DWORD(text.size()), &n, nullptr);
    CloseHandle(f);
}

std::string ReadFileText(const std::wstring& path) {
    std::string text;
    HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, 0, nullptr);
    if (f == INVALID_HANDLE_VALUE) return text;
    char buffer[65536];
    DWORD n = 0;
    while (ReadFile(f, buffer, sizeof buffer, &n, nullptr) && n) text.append(buffer, n);
    CloseHandle(f);
    return text;
}

size_t Count(const std::string& text, const char* needle) {
    size_t n = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) ++n;
    return n;
}

struct Api {
    HMODULE module = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxConfigure configure = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    bool Load() {
        module = LoadLibraryW((g_dir + L"\\" + g_dllName).c_str());
        if (!module) return false;
        create = reinterpret_cast<PfnFfxCreateContext>(reinterpret_cast<void*>(GetProcAddress(module, "ffxCreateContext")));
        destroy = reinterpret_cast<PfnFfxDestroyContext>(reinterpret_cast<void*>(GetProcAddress(module, "ffxDestroyContext")));
        configure = reinterpret_cast<PfnFfxConfigure>(reinterpret_cast<void*>(GetProcAddress(module, "ffxConfigure")));
        query = reinterpret_cast<PfnFfxQuery>(reinterpret_cast<void*>(GetProcAddress(module, "ffxQuery")));
        dispatch = reinterpret_cast<PfnFfxDispatch>(reinterpret_cast<void*>(GetProcAddress(module, "ffxDispatch")));
        return create && destroy && configure && query && dispatch;
    }
};

std::string FakeLog() {
    HMODULE fake = GetModuleHandleW(L"amd_fidelityfx_dx12.wronggpu-original.dll");
    if (!fake) return "(fake AMD DLL not loaded)";
    auto fn = reinterpret_cast<const char* (*)()>(reinterpret_cast<void*>(GetProcAddress(fake, "fakeLog")));
    return fn ? fn() : "(no fakeLog)";
}

// ---------------------------------------------------------------------------------------------------------------
// Minimal D3D12 helpers.

struct Gpu {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    // A second list of the "game", recorded and submitted independently (split and trace modes).
    ID3D12CommandAllocator* allocatorB = nullptr;
    ID3D12GraphicsCommandList* listB = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE event = nullptr;
    UINT64 value = 0;
    std::vector<IUnknown*> keep;
    bool debugLayer = false;

    bool Init() {
        HMODULE d3d12 = LoadLibraryW(L"d3d12.dll");
#if defined(_MSC_VER)
        // The D3D12 debug layer (Windows "Graphics Tools"), unless WG_TEST_D3D12_DEBUG=0; CheckDebugMessages() fails the
        // test on its errors.
        char env[8] = "1";
        GetEnvironmentVariableA("WG_TEST_D3D12_DEBUG", env, sizeof env);
        auto getDebug = d3d12 ? reinterpret_cast<PFN_D3D12_GET_DEBUG_INTERFACE>(reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12GetDebugInterface"))) : nullptr;
        ID3D12Debug* debug = nullptr;
        if (env[0] != '0' && getDebug && SUCCEEDED(getDebug(IID_PPV_ARGS(&debug))) && debug) {
            debug->EnableDebugLayer();
            debug->Release();
            debugLayer = true;
        }
        std::printf("D3D12 debug layer: %s\n", debugLayer ? "on" : "not available (WG_TEST_D3D12_DEBUG=0 or Graphics Tools not installed)");
#else
        std::printf("D3D12 debug layer: not used (MinGW build; the MSVC build enables it)\n");
#endif
        auto createDevice = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice"))) : nullptr;
        if (!createDevice || FAILED(createDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) return false;
#if defined(_MSC_VER)
        if (debugLayer) {
            ID3D12InfoQueue* info = nullptr;
            if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&info))) && info) {
                info->SetMessageCountLimit(4096);
                info->Release();
            }
        }
#endif
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)))) return false;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)))) return false;
        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list)))) return false;
        list->Close();
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocatorB)))) return false;
        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocatorB, nullptr, IID_PPV_ARGS(&listB)))) return false;
        listB->Close();
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return false;
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return true;
    }
    void Begin() {
        allocator->Reset();
        list->Reset(allocator, nullptr);
    }
    bool SubmitAndWait() {
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        return Wait();
    }
    // Waits until the queue has run everything submitted so far.
    bool Wait() {
        queue->Signal(fence, ++value);
        fence->SetEventOnCompletion(value, event);
        return WaitForSingleObject(event, 60000) == WAIT_OBJECT_0;
    }
    // Debug-layer messages: errors fail the test, warnings are printed (MSVC builds with the debug layer only).
    void CheckDebugMessages() {
#if defined(_MSC_VER)
        if (!debugLayer) return;
        ID3D12InfoQueue* info = nullptr;
        if (FAILED(device->QueryInterface(IID_PPV_ARGS(&info))) || !info) return;
        const UINT64 n = info->GetNumStoredMessages();
        int errors = 0;
        for (UINT64 i = 0; i < n; ++i) {
            SIZE_T size = 0;
            info->GetMessage(i, nullptr, &size);
            std::vector<char> buffer(size);
            auto* m = reinterpret_cast<D3D12_MESSAGE*>(buffer.data());
            if (FAILED(info->GetMessage(i, m, &size))) continue;
            const bool error = m->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION || m->Severity == D3D12_MESSAGE_SEVERITY_ERROR;
            if (error) ++errors;
            if (m->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
                std::printf("D3D12 debug layer %s: %s\n", error ? "ERROR" : "warning", m->pDescription);
        }
        info->Release();
        std::printf("D3D12 debug layer: %d errors\n", errors);
        CHECK(errors == 0);
#endif
    }
    UINT16 mips = 1;  // mode upscale_mips (bug hunt F): the game's textures with as many mips as their size allows
    ID3D12Resource* Texture(DXGI_FORMAT format, UINT w, UINT h, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = w;
        d.Height = h;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        if (mips > 1 && !(flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) {
            UINT16 most = 1;
            for (UINT m = w > h ? w : h; m > 1; m >>= 1) ++most;
            d.MipLevels = mips < most ? mips : most;
        }
        d.Format = format;
        d.SampleDesc.Count = 1;
        d.Flags = flags;
        ID3D12Resource* r = nullptr;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
        keep.push_back(r);
        return r;
    }
    ID3D12Resource* Buffer(UINT64 size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = type;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = size;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource* r = nullptr;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
        keep.push_back(r);
        return r;
    }
    static D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
        D3D12_RESOURCE_BARRIER t{};
        t.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        t.Transition.pResource = r;
        t.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        t.Transition.StateBefore = a;
        t.Transition.StateAfter = b;
        return t;
    }
    // Uploads tightly packed rows into a texture in COPY_DEST state and leaves it in `after` (recorded, not run).
    void Upload(ID3D12Resource* texture, const void* data, UINT pixelBytes, D3D12_RESOURCE_STATES after) {
        const D3D12_RESOURCE_DESC d = texture->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT rows = 0;
        UINT64 rowBytes = 0, total = 0;
        device->GetCopyableFootprints(&d, 0, 1, 0, &fp, &rows, &rowBytes, &total);
        ID3D12Resource* staging = Buffer(total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        unsigned char* p = nullptr;
        staging->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (UINT y = 0; y < rows; ++y)
            std::memcpy(p + fp.Offset + UINT64(y) * fp.Footprint.RowPitch, static_cast<const unsigned char*>(data) + UINT64(y) * d.Width * pixelBytes,
                        size_t(d.Width) * pixelBytes);
        staging->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = texture;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource = staging;
        src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint = fp;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        if (after != D3D12_RESOURCE_STATE_COPY_DEST) {
            const D3D12_RESOURCE_BARRIER b = Transition(texture, D3D12_RESOURCE_STATE_COPY_DEST, after);
            list->ResourceBarrier(1, &b);
        }
    }
    // Copies a texture (in `state`) to a readback buffer; Read() returns the data after SubmitAndWait.
    struct Pending {
        ID3D12Resource* buffer = nullptr;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT rows = 0, width = 0, pixelBytes = 0;
    };
    Pending CopyOut(ID3D12Resource* texture, UINT pixelBytes, D3D12_RESOURCE_STATES state) {
        Pending r;
        const D3D12_RESOURCE_DESC d = texture->GetDesc();
        UINT64 rowBytes = 0, total = 0;
        device->GetCopyableFootprints(&d, 0, 1, 0, &r.fp, &r.rows, &rowBytes, &total);
        r.buffer = Buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        r.width = UINT(d.Width);
        r.pixelBytes = pixelBytes;
        D3D12_RESOURCE_BARRIER b = Transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        if (state != D3D12_RESOURCE_STATE_COPY_SOURCE) list->ResourceBarrier(1, &b);
        D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
        dst.pResource = r.buffer;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = r.fp;
        src.pResource = texture;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        b = Transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
        if (state != D3D12_RESOURCE_STATE_COPY_SOURCE) list->ResourceBarrier(1, &b);
        return r;
    }
    std::vector<float> Read(const Pending& r) {
        std::vector<float> out(size_t(r.width) * r.rows * r.pixelBytes / 4);
        unsigned char* p = nullptr;
        r.buffer->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (UINT y = 0; y < r.rows; ++y)
            std::memcpy(reinterpret_cast<unsigned char*>(out.data()) + size_t(y) * r.width * r.pixelBytes,
                        p + r.fp.Offset + UINT64(y) * r.fp.Footprint.RowPitch, size_t(r.width) * r.pixelBytes);
        r.buffer->Unmap(0, nullptr);
        return out;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Scene and CPU references.

// Texture sizes: render textures larger than the render region, output texture larger than the output region.
constexpr UINT kTexW = 40, kTexH = 30, kRW = 32, kRH = 24;
constexpr UINT kOutTexW = 70, kOutTexH = 50, kOW = 64, kOH = 48;
constexpr float kSentinel = 7.0f;

struct Float4 {
    float x, y, z, w;
};

// A smooth test scene (render-pixel units of the unjittered screen).
Float4 Scene(double x, double y) {
    return {float(0.5 + 0.4 * std::sin(0.37 * x + 0.11 * y)), float(0.5 + 0.4 * std::cos(0.23 * y - 0.05 * x)),
            float(0.25 + 0.004 * (x + y)), 1.0f};
}

// The game's render of frame k: FSR's jitterOffset j shifts the image by +j (AMD's projection example), and the
// content moved by d, so pixel center c shows the scene point c - j - d.
std::vector<Float4> Render(float jx, float jy, float dx, float dy) {
    std::vector<Float4> c(kTexW * kTexH, Float4{-5, -5, -5, 1});  // outside the render region: garbage
    for (UINT y = 0; y < kRH; ++y)
        for (UINT x = 0; x < kRW; ++x) c[y * kTexW + x] = Scene(x + 0.5 - jx - dx, y + 0.5 - jy - dy);
    return c;
}

Float4 Load(const std::vector<Float4>& c, int x, int y) {
    x = std::min(std::max(x, 0), int(kRW) - 1);
    y = std::min(std::max(y, 0), int(kRH) - 1);
    return c[y * kTexW + x];
}

Float4 Mix(Float4 a, Float4 b, double t) {
    return {float(a.x + (b.x - a.x) * t), float(a.y + (b.y - a.y) * t), float(a.z + (b.z - a.z) * t), float(a.w + (b.w - a.w) * t)};
}

Float4 BilinearRef(const std::vector<Float4>& c, double tx, double ty) {
    const double qx = tx - 0.5, qy = ty - 0.5;
    const double bx = std::floor(qx), by = std::floor(qy);
    const int ix = int(bx), iy = int(by);
    const double fx = qx - bx, fy = qy - by;
    return Mix(Mix(Load(c, ix, iy), Load(c, ix + 1, iy), fx), Mix(Load(c, ix, iy + 1), Load(c, ix + 1, iy + 1), fx), fy);
}

double Lanczos2(double x) {
    x = std::fabs(x);
    if (x < 1e-5) return 1.0;
    if (x >= 2.0) return 0.0;
    const double px = 3.14159265358979 * x;
    return 2.0 * std::sin(px) * std::sin(0.5 * px) / (px * px);
}

Float4 LanczosRef(const std::vector<Float4>& c, double tx, double ty) {
    const double qx = tx - 0.5, qy = ty - 0.5;
    const double bx = std::floor(qx), by = std::floor(qy);
    const int ix = int(bx), iy = int(by);
    const double fx = qx - bx, fy = qy - by;
    double s[4] = {0, 0, 0, 0}, ws = 0;
    float lo[4] = {1e30f, 1e30f, 1e30f, 1e30f}, hi[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    for (int y = -1; y <= 2; ++y)
        for (int x = -1; x <= 2; ++x) {
            const double w = Lanczos2(x - fx) * Lanczos2(y - fy);
            const Float4 v = Load(c, ix + x, iy + y);
            const float vv[4] = {v.x, v.y, v.z, v.w};
            for (int k = 0; k < 4; ++k) s[k] += w * vv[k];
            ws += w;
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1)
                for (int k = 0; k < 4; ++k) {
                    lo[k] = std::min(lo[k], vv[k]);
                    hi[k] = std::max(hi[k], vv[k]);
                }
        }
    float r[4];
    for (int k = 0; k < 4; ++k) r[k] = std::min(std::max(float(s[k] / ws), lo[k]), hi[k]);
    return {r[0], r[1], r[2], r[3]};
}

double Abs(double a, double b) { return std::fabs(a - b); }

FfxApiResource Resource(ID3D12Resource* r, uint32_t ffxFormat, UINT w, UINT h, uint32_t state) {
    FfxApiResource x{};
    x.resource = r;
    x.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    x.description.format = ffxFormat;
    x.description.width = w;
    x.description.height = h;
    x.description.depth = 1;
    x.description.mipCount = 1;
    x.state = state;
    return x;
}

// ---------------------------------------------------------------------------------------------------------------

int RunUpscale(Api& api, Gpu& gpu, bool lanczos) {
    const float jx = 0.3125f, jy = -0.2f;  // FSR jitterOffset of this frame
    std::vector<Float4> color = Render(jx, jy, 0, 0);
    ID3D12Resource* colorTex = gpu.Texture(DXGI_FORMAT_R32G32B32A32_TYPELESS, kTexW, kTexH, D3D12_RESOURCE_FLAG_NONE,
                                           D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* depthTex = gpu.Texture(DXGI_FORMAT_R32_FLOAT, kTexW, kTexH, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* outTex = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, kOutTexW, kOutTexH,
                                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS | D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                         D3D12_RESOURCE_STATE_COPY_DEST);
    CHECK(colorTex && depthTex && outTex);
    if (!colorTex || !depthTex || !outTex) return 1;
    std::vector<float> depth(kTexW * kTexH, 0.5f);
    std::vector<Float4> sentinel(kOutTexW * kOutTexH, Float4{kSentinel, kSentinel, kSentinel, kSentinel});
    // The game leaves color readable by pixel and compute shaders and the output as a render target (lanczos: UAV).
    const D3D12_RESOURCE_STATES outState = lanczos ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_RENDER_TARGET;
    gpu.Begin();
    gpu.Upload(colorTex, color.data(), 16, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Upload(depthTex, depth.data(), 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Upload(outTex, sentinel.data(), 16, outState);
    CHECK(gpu.SubmitAndWait());

    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = gpu.device;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &backend.header;
    create.flags = FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    create.maxRenderSize = {kTexW, kTexH};
    create.maxUpscaleSize = {kOW, kOH};
    ffxContext ctx = nullptr;
    CHECK(api.create(&ctx, &create.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(ctx != nullptr);
    ffxQueryGetProviderVersion pv{};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    CHECK(api.query(&ctx, &pv.header) == FFX_API_RETURN_OK);
    CHECK(pv.versionName && std::strstr(pv.versionName, "WrongGPU") != nullptr);
    if (pv.versionName && !std::strstr(pv.versionName, "WrongGPU")) {
        std::printf("upscaler context is not WrongGPU's (%s); see wronggpu-fsr.log\n", pv.versionName);
        return 1;
    }

    gpu.Begin();
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = gpu.list;
    d.color = Resource(colorTex, FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    d.depth = Resource(depthTex, FFX_API_SURFACE_FORMAT_R32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = Resource(outTex, FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT, kOutTexW, kOutTexH,
                        lanczos ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_RENDER_TARGET);
    d.jitterOffset = {jx, jy};
    d.motionVectorScale = {float(kRW), float(kRH)};
    d.renderSize = {kRW, kRH};
    d.upscaleSize = {kOW, kOH};
    d.preExposure = 1.0f;
    d.frameTimeDelta = 16.7f;
    CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    Gpu::Pending pending = gpu.CopyOut(outTex, 16, outState);
    CHECK(gpu.SubmitAndWait());
    std::vector<float> out = gpu.Read(pending);

    // 1) Same arithmetic as the shader (texel position = render position - network jitter = + FSR jitterOffset).
    // 2) The true scene: the jitter is compensated, so only the filter's interpolation error remains; with the
    //    opposite jitter sign the error would be about twice the jitter times the scene gradient.
    double maxRef = 0, maxTruth = 0, maxWrongSign = 0;
    int untouched = 0, touchedOutside = 0;
    for (UINT y = 0; y < kOutTexH; ++y)
        for (UINT x = 0; x < kOutTexW; ++x) {
            const float* p = &out[(size_t(y) * kOutTexW + x) * 4];
            if (x >= kOW || y >= kOH) {
                touchedOutside += p[0] != kSentinel;
                continue;
            }
            untouched += p[0] == kSentinel;
            const double ux = (x + 0.5) * kRW / kOW, uy = (y + 0.5) * kRH / kOH;
            const Float4 ref = lanczos ? LanczosRef(color, ux + jx, uy + jy) : BilinearRef(color, ux + jx, uy + jy);
            maxRef = std::max({maxRef, Abs(p[0], ref.x), Abs(p[1], ref.y), Abs(p[2], ref.z)});
            if (x >= 4 && y >= 4 && x < kOW - 4 && y < kOH - 4) {
                const Float4 truth = Scene(ux, uy);
                maxTruth = std::max({maxTruth, Abs(p[0], truth.x), Abs(p[1], truth.y)});
                const Float4 wrong = BilinearRef(color, ux - jx, uy - jy);
                maxWrongSign = std::max({maxWrongSign, Abs(wrong.x, truth.x), Abs(wrong.y, truth.y)});
            }
        }
    std::printf("%s: max |GPU - CPU reference| = %.3g, max |GPU - scene| = %.3g (wrong jitter sign would give %.3g)\n",
                lanczos ? "lanczos" : "bilinear", maxRef, maxTruth, maxWrongSign);
    CHECK(maxRef < 2e-4);
    CHECK(maxTruth < 0.02);
    CHECK(maxWrongSign > 2.0 * maxTruth);
    CHECK(untouched == 0);
    CHECK(touchedOutside == 0);

    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
    CHECK(ctx == nullptr);
    return 0;
}

// Two frames of a moving scene through the warp-difference view: |current - previous warped by the motion|.
int RunWarp(Api& api, Gpu& gpu, bool expectLarge) {
    const float j0x = 0.25f, j0y = -0.2f, j1x = -0.3f, j1y = 0.15f;
    const float d1x = 1.5f, d1y = -1.0f;  // content movement from frame 0 to frame 1, render pixels
    std::vector<Float4> c0 = Render(j0x, j0y, 0, 0), c1 = Render(j1x, j1y, d1x, d1y);
    // Unjittered motion of frame 1, previous position - current position = -d1; the game stores it in UV units
    // (motionVectorScale = render size), as AMD's NDC example does.
    struct Float2 {
        float x, y;
    };
    std::vector<Float2> mv(kTexW * kTexH, Float2{-d1x / kRW, -d1y / kRH});
    ID3D12Resource* color[2];
    for (auto& c : color)
        c = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, kTexW, kTexH, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* motion = gpu.Texture(DXGI_FORMAT_R32G32_FLOAT, kTexW, kTexH, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* outTex = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, kOW, kOH, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    gpu.Begin();
    gpu.Upload(color[0], c0.data(), 16, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Upload(color[1], c1.data(), 16, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Upload(motion, mv.data(), 8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    CHECK(gpu.SubmitAndWait());

    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = gpu.device;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &backend.header;
    create.maxRenderSize = {kTexW, kTexH};
    create.maxUpscaleSize = {kOW, kOH};
    ffxContext ctx = nullptr;
    CHECK(api.create(&ctx, &create.header, nullptr) == FFX_API_RETURN_OK);

    std::vector<float> out;
    for (int frame = 0; frame < 2; ++frame) {
        gpu.Begin();
        ffxDispatchDescUpscale d{};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = gpu.list;
        d.color = Resource(color[frame], FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.motionVectors = Resource(motion, FFX_API_SURFACE_FORMAT_R32G32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.output = Resource(outTex, FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT, kOW, kOH, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = frame ? FfxApiFloatCoords2D{j1x, j1y} : FfxApiFloatCoords2D{j0x, j0y};
        d.motionVectorScale = {float(kRW), float(kRH)};
        d.renderSize = {kRW, kRH};
        d.preExposure = 1.0f;
        d.reset = frame == 0;
        CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
        Gpu::Pending pending = gpu.CopyOut(outTex, 16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CHECK(gpu.SubmitAndWait());
        out = gpu.Read(pending);
    }
    // Inside a margin (the previous frame does not cover the borders), the difference is only filter error.
    double sum = 0, maxDiff = 0;
    int n = 0;
    for (UINT y = 8; y < kOH - 8; ++y)
        for (UINT x = 8; x < kOW - 8; ++x) {
            const float* p = &out[(size_t(y) * kOW + x) * 4];
            const double v = std::max(p[0], p[1]);
            sum += v;
            maxDiff = std::max(maxDiff, v);
            ++n;
        }
    const double mean = sum / n;
    std::printf("warpdiff (motion_sign %s): mean difference %.4g, max %.4g\n", expectLarge ? "-1" : "+1", mean, maxDiff);
    if (expectLarge) {
        CHECK(mean > 0.1);
    } else {
        CHECK(maxDiff < 0.03);
    }
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
    return 0;
}

// The debug views (motion, depth, jitter) and an sRGB output: the dispatch succeeds and writes plausible values.
int RunViewSmoke(Api& api, Gpu& gpu, const std::string& mode) {
    constexpr UINT W = 96, H = 72, OW = 192, OH = 144;
    const bool srgb = mode == "srgb";
    const float jx = 0.25f, jy = -0.125f;
    std::vector<Float4> color(W * H);
    for (UINT y = 0; y < H; ++y)
        for (UINT x = 0; x < W; ++x) color[y * W + x] = Scene(x + 0.5, y + 0.5);
    struct Float2 {
        float x, y;
    };
    std::vector<Float2> mv(W * H, Float2{4.0f / W, -2.0f / H});  // 4 and -2 render pixels
    std::vector<float> depth(W * H);
    for (UINT i = 0; i < W * H; ++i) depth[i] = float(i % W) / W;
    ID3D12Resource* colorTex = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* mvTex = gpu.Texture(DXGI_FORMAT_R32G32_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* depthTex = gpu.Texture(DXGI_FORMAT_R32_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* outTex = gpu.Texture(srgb ? DXGI_FORMAT_R8G8B8A8_TYPELESS : DXGI_FORMAT_R32G32B32A32_FLOAT, OW, OH,
                                         D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    CHECK(colorTex && mvTex && depthTex && outTex);
    if (!colorTex || !mvTex || !depthTex || !outTex) return 1;
    gpu.Begin();
    gpu.Upload(colorTex, color.data(), 16, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Upload(mvTex, mv.data(), 8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Upload(depthTex, depth.data(), 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    CHECK(gpu.SubmitAndWait());

    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = gpu.device;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &backend.header;
    create.flags = FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
    create.maxRenderSize = {W, H};
    create.maxUpscaleSize = {OW, OH};
    ffxContext ctx = nullptr;
    CHECK(api.create(&ctx, &create.header, nullptr) == FFX_API_RETURN_OK);
    gpu.Begin();
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = gpu.list;
    d.color = Resource(colorTex, FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = Resource(depthTex, FFX_API_SURFACE_FORMAT_R32_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.motionVectors = Resource(mvTex, FFX_API_SURFACE_FORMAT_R32G32_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = Resource(outTex, srgb ? FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB : FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT, OW, OH,
                        FFX_API_RESOURCE_STATE_COMMON);
    d.jitterOffset = {jx, jy};
    d.motionVectorScale = {float(W), float(H)};
    d.renderSize = {W, H};
    d.preExposure = 1.0f;
    CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    Gpu::Pending pending = gpu.CopyOut(outTex, srgb ? 4 : 16, D3D12_RESOURCE_STATE_COMMON);
    CHECK(gpu.SubmitAndWait());
    std::vector<float> out = gpu.Read(pending);
    auto px = [&](UINT x, UINT y) { return &out[(size_t(y) * OW + x) * 4]; };
    if (mode == "view_motion") {
        // red = 0.5 + 4 px * gain(4) / 64 = 0.75, green = 0.5 - 2 * 4 / 64 = 0.375
        CHECK(std::fabs(px(100, 70)[0] - 0.75f) < 1e-3 && std::fabs(px(100, 70)[1] - 0.375f) < 1e-3);
    } else if (mode == "view_depth") {
        // depth grows to the right; inverted depth: near (large) is bright
        CHECK(px(OW - 4, 70)[0] > px(4, 70)[0] && px(4, 70)[0] >= 0.0f && px(OW - 4, 70)[0] <= 1.0f);
    } else if (mode == "view_jitter") {
        // the dot sits at (64, 64) + round(network jitter * 128) = (64 + 32, 64 - 16) (j = +jitterOffset)
        CHECK(px(96, 48)[0] > 0.9f && px(96, 48)[1] < 0.2f);
        CHECK(px(96 + 32, 80)[0] == px(96 + 32, 80)[0]);  // finite outside the box
    } else if (srgb) {
        const auto* bytes = reinterpret_cast<const unsigned char*>(out.data());
        int worst = 0;
        for (UINT y = 4; y < OH - 4; ++y)
            for (UINT x = 4; x < OW - 4; ++x) {
                const double ux = (x + 0.5) * W / OW + jx, uy = (y + 0.5) * H / OH + jy;
                const double qx = ux - 0.5, qy = uy - 0.5;
                const int ix = int(std::floor(qx)), iy = int(std::floor(qy));
                const double fx = qx - ix, fy = qy - iy;
                auto at = [&](int a, int b) { return color[std::min(std::max(b, 0), int(H) - 1) * W + std::min(std::max(a, 0), int(W) - 1)]; };
                const Float4 ref = Mix(Mix(at(ix, iy), at(ix + 1, iy), fx), Mix(at(ix, iy + 1), at(ix + 1, iy + 1), fx), fy);
                const double c = std::min(std::max(double(ref.x), 0.0), 1.0);
                const double e = c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
                worst = std::max(worst, std::abs(int(bytes[(size_t(y) * OW + x) * 4]) - int(std::lround(e * 255.0))));
            }
        std::printf("srgb: worst difference %d/255\n", worst);
        CHECK(worst <= 1);
    }
    for (size_t i = 0; !srgb && i < out.size(); ++i) CHECK(std::isfinite(out[i]) || (std::printf("non-finite at %zu\n", i), false));
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// FSR 3.0 component DLLs.

namespace f30 = wgfsr::fsr30;

std::string FakeLogOf(const wchar_t* module) {
    HMODULE m = GetModuleHandleW(module);
    if (!m) return "(not loaded)\n";
    auto fn = reinterpret_cast<const char* (*)()>(reinterpret_cast<void*>(GetProcAddress(m, "fakeLog")));
    return fn ? fn() : "(no fakeLog)";
}

template <class T>
T Export(HMODULE m, const char* name) {
    T f = reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(m, name)));
    if (!f) std::printf("FAIL: export %s missing\n", name), ++g_failures;
    return f;
}

f30::Resource Res30(ID3D12Resource* r, uint32_t format, UINT w, UINT h, uint32_t state) {
    f30::Resource x{};
    x.resource = r;
    x.description.type = 2;  // TEXTURE2D
    x.description.format = format;
    x.description.width = w;
    x.description.height = h;
    x.description.depth = 1;
    x.description.mipCount = 1;
    x.state = state;
    return x;
}

struct Scene30 {
    std::vector<Float4> color;
    ID3D12Resource* colorTex = nullptr;
    ID3D12Resource* depthTex = nullptr;
    ID3D12Resource* outTex = nullptr;
    float jx = 0.3125f, jy = -0.2f;
};

bool MakeScene30(Gpu& gpu, Scene30& s) {
    s.color = Render(s.jx, s.jy, 0, 0);
    s.colorTex = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, kTexW, kTexH, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    s.depthTex = gpu.Texture(DXGI_FORMAT_R32_FLOAT, kTexW, kTexH, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    s.outTex = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, kOutTexW, kOutTexH, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COPY_DEST);
    if (!s.colorTex || !s.depthTex || !s.outTex) return false;
    std::vector<float> depth(kTexW * kTexH, 0.5f);
    std::vector<Float4> sentinel(kOutTexW * kOutTexH, Float4{kSentinel, kSentinel, kSentinel, kSentinel});
    gpu.Begin();
    gpu.Upload(s.colorTex, s.color.data(), 16, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Upload(s.depthTex, depth.data(), 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    gpu.Upload(s.outTex, sentinel.data(), 16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return gpu.SubmitAndWait();
}

// The output region must hold the stand-in's bilinear upscale with jitter compensation (as in mode "upscale").
void CheckScene30(Gpu& gpu, Scene30& s, const char* label) {
    gpu.Begin();
    Gpu::Pending pending = gpu.CopyOut(s.outTex, 16, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CHECK(gpu.SubmitAndWait());
    std::vector<float> out = gpu.Read(pending);
    double maxRef = 0;
    int untouched = 0;
    for (UINT y = 0; y < kOH; ++y)
        for (UINT x = 0; x < kOW; ++x) {
            const float* p = &out[(size_t(y) * kOutTexW + x) * 4];
            untouched += p[0] == kSentinel;
            const double ux = (x + 0.5) * kRW / kOW, uy = (y + 0.5) * kRH / kOH;
            const Float4 ref = BilinearRef(s.color, ux + s.jx, uy + s.jy);
            maxRef = std::max({maxRef, Abs(p[0], ref.x), Abs(p[1], ref.y), Abs(p[2], ref.z)});
        }
    std::printf("%s: max |GPU - CPU reference| = %.3g\n", label, maxRef);
    CHECK(maxRef < 2e-4);
    CHECK(untouched == 0);
}

f30::UpscalerDispatchDescription UpDispatch30(Gpu& gpu, Scene30& s, bool shared) {
    static char dummy[3];
    f30::UpscalerDispatchDescription d{};
    d.commandList = gpu.list;
    d.color = Res30(s.colorTex, f30::SF_R32G32B32A32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = Res30(s.depthTex, f30::SF_R32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = Res30(s.outTex, f30::SF_R32G32B32A32_FLOAT, kOutTexW, kOutTexH, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    if (shared) {  // frame-generation resources: only the fake AMD upscaler sees them
        d.dilatedDepth.resource = &dummy[0];
        d.dilatedMotionVectors.resource = &dummy[1];
        d.reconstructedPrevNearestDepth.resource = &dummy[2];
    }
    d.jitterOffset = {s.jx, s.jy};
    d.motionVectorScale = {float(kRW), float(kRH)};
    d.renderSize = {kRW, kRH};
    d.preExposure = 1.0f;
    d.frameTimeDelta = 16.7f;
    return d;
}

void RunFsr30Upscaler(const std::string& mode, Gpu* gpu) {
    HMODULE m = LoadLibraryW((g_dir + L"\\ffx_fsr3upscaler_x64.dll").c_str());
    CHECK(m != nullptr);
    if (!m) return;
    auto create = Export<f30::ErrorCode (*)(void*, const f30::UpscalerContextDescription*)>(m, "ffxFsr3UpscalerContextCreate");
    auto dispatch = Export<f30::ErrorCode (*)(void*, const f30::UpscalerDispatchDescription*)>(m, "ffxFsr3UpscalerContextDispatch");
    auto destroy = Export<f30::ErrorCode (*)(void*)>(m, "ffxFsr3UpscalerContextDestroy");
    auto phases = Export<int32_t (*)(int32_t, int32_t)>(m, "ffxFsr3UpscalerGetJitterPhaseCount");
    auto jitter = Export<f30::ErrorCode (*)(float*, float*, int32_t, int32_t)>(m, "ffxFsr3UpscalerGetJitterOffset");
    auto isNull = Export<bool (*)(f30::Resource)>(m, "ffxFsr3UpscalerResourceIsNull");  // by value, as AMD declares it
    if (!create || !dispatch || !destroy || !phases || !jitter || !isNull) return;
    std::vector<uint32_t> ctx(f30::kUpscalerContextBytes / 4);
    f30::UpscalerContextDescription cd{};
    cd.flags = f30::FLAG_DEPTH_INVERTED;
    cd.maxRenderSize = {kTexW, kTexH};
    cd.displaySize = {kOW, kOH};
    CHECK(create(ctx.data(), &cd) == f30::OK);
    CHECK(phases(kRW, kOW) == 77);  // answered by the fake AMD DLL
    float x = 0, y = 0;
    CHECK(jitter(&x, &y, 0, 77) == f30::OK && x == 0.125f && y == -0.25f);
    char something;
    f30::Resource r{};
    r.resource = &something;
    CHECK(!isNull(r));
    r.resource = nullptr;
    CHECK(isNull(r));

    if (mode == "fsr30_log") {
        f30::UpscalerDispatchDescription d{};
        d.renderSize = {kRW, kRH};
        CHECK(dispatch(ctx.data(), &d) == f30::OK);
    } else {
        Scene30 s;
        CHECK(MakeScene30(*gpu, s));
        gpu->Begin();
        f30::UpscalerDispatchDescription d = UpDispatch30(*gpu, s, mode == "fsr30_upscaler_shared");
        CHECK(dispatch(ctx.data(), &d) == f30::OK);
        CHECK(gpu->SubmitAndWait());
        CheckScene30(*gpu, s, mode.c_str());
    }
    CHECK(destroy(ctx.data()) == f30::OK);
    const std::string fake = FakeLogOf(L"ffx_fsr3upscaler_x64.wronggpu-original.dll");
    std::printf("fake AMD ffx_fsr3upscaler saw:\n%s", fake.c_str());
    CHECK(Count(fake, "upscaler.create") == 1 && Count(fake, "upscaler.destroy") == 1);
    CHECK(Count(fake, "upscaler.isnull ctx=0000000000000000 resource") == 1 && Count(fake, "upscaler.isnull ctx=0000000000000000 null") == 1);
    if (mode == "fsr30_upscaler") CHECK(Count(fake, "upscaler.dispatch") == 0);
    else if (mode == "fsr30_upscaler_shared") CHECK(Count(fake, "upscaler.dispatch") == 1 && Count(fake, "shared=1") == 1);
    else CHECK(Count(fake, "upscaler.dispatch") == 1);
}

void RunFsr30Fsr3(const std::string& mode, Gpu& gpu) {
    const bool upscalingOnly = mode == "fsr30_fsr3_upscaling_only";  // Cyberpunk 2077 with frame generation off
    const bool inner = mode == "fsr30_fsr3_inner" || upscalingOnly;
    SetEnvironmentVariableA("WG_FAKE_FSR3_CALLS_UPSCALER", inner ? "1" : "0");
    HMODULE m = LoadLibraryW((g_dir + L"\\ffx_fsr3_x64.dll").c_str());
    CHECK(m != nullptr);
    if (!m) return;
    auto create = Export<f30::ErrorCode (*)(void*, f30::Fsr3ContextDescription*)>(m, "ffxFsr3ContextCreate");
    auto dispatch = Export<f30::ErrorCode (*)(void*, const f30::Fsr3DispatchUpscaleDescription*)>(m, "ffxFsr3ContextDispatchUpscale");
    auto configure = Export<f30::ErrorCode (*)(void*, const f30::FrameGenerationConfig*)>(m, "ffxFsr3ConfigureFrameGeneration");
    auto destroy = Export<f30::ErrorCode (*)(void*)>(m, "ffxFsr3ContextDestroy");
    if (!create || !dispatch || !configure || !destroy) return;
    std::vector<uint32_t> ctx(f30::kFsr3ContextBytes / 4);
    f30::Fsr3ContextDescription cd{};
    cd.flags = f30::FLAG_DEPTH_INVERTED | (upscalingOnly ? f30::FSR3_FLAG_UPSCALING_ONLY : 0u);
    cd.maxRenderSize = {kTexW, kTexH};
    cd.upscaleOutputSize = {kOW, kOH};
    cd.displaySize = {kOW, kOH};
    CHECK(create(ctx.data(), &cd) == f30::OK);
    f30::FrameGenerationConfig fg{};
    fg.frameGenerationEnabled = true;
    CHECK(configure(ctx.data(), &fg) == f30::OK);
    Scene30 s;
    CHECK(MakeScene30(gpu, s));
    gpu.Begin();
    f30::Fsr3DispatchUpscaleDescription d{};
    d.commandList = gpu.list;
    d.color = Res30(s.colorTex, f30::SF_R32G32B32A32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = Res30(s.depthTex, f30::SF_R32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.upscaleOutput = Res30(s.outTex, f30::SF_R32G32B32A32_FLOAT, kOutTexW, kOutTexH, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = {s.jx, s.jy};
    d.motionVectorScale = {float(kRW), float(kRH)};
    d.renderSize = {kRW, kRH};
    d.preExposure = 1.0f;
    CHECK(dispatch(ctx.data(), &d) == f30::OK);
    CHECK(gpu.SubmitAndWait());
    CheckScene30(gpu, s, mode.c_str());
    CHECK(destroy(ctx.data()) == f30::OK);
    const std::string fsr3 = FakeLogOf(L"ffx_fsr3_x64.wronggpu-original.dll");
    const std::string up = FakeLogOf(L"ffx_fsr3upscaler_x64.wronggpu-original.dll");
    std::printf("fake AMD ffx_fsr3 saw:\n%sfake AMD ffx_fsr3upscaler saw:\n%s", fsr3.c_str(), up.c_str());
    CHECK(Count(fsr3, "fsr3.create") == 1 && Count(fsr3, "fsr3.dispatchUpscale") == 1 && Count(fsr3, "fsr3.destroy") == 1);
    CHECK(Count(fsr3, "fsr3.configureFrameGeneration") == 1 && Count(fsr3, "enabled=1") == 1);
    if (upscalingOnly) CHECK(Count(up, "upscaler.create") == 1 && Count(up, "upscaler.dispatch") == 0);  // AMD's upscaler skipped
    else if (inner) CHECK(Count(up, "upscaler.create") == 1 && Count(up, "upscaler.dispatch") == 1 && Count(up, "shared=1") == 1);
    else CHECK(Count(up, "upscaler.") == 0);
}

void RunForward(Api& api) {
    // Frame generation: create, configure, dispatch, destroy go to AMD's DLL with AMD's context value.
    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    ffxCreateContextDescFrameGeneration fg{};
    fg.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    fg.header.pNext = &backend.header;
    fg.displaySize = {kOW, kOH};
    fg.maxRenderSize = {kRW, kRH};
    ffxContext fgctx = nullptr;
    CHECK(api.create(&fgctx, &fg.header, nullptr) == FFX_API_RETURN_OK);
    CHECK(reinterpret_cast<uintptr_t>(fgctx) >= 0xAD0000);  // the fake AMD DLL's context value, unwrapped
    ffxConfigureDescFrameGeneration cfg{};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    CHECK(api.configure(&fgctx, &cfg.header) == FFX_API_RETURN_OK);
    ffxDispatchDescFrameGenerationPrepare prep{};
    prep.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
    CHECK(api.dispatch(&fgctx, &prep.header) == FFX_API_RETURN_OK);
    CHECK(api.destroy(&fgctx, nullptr) == FFX_API_RETURN_OK);

    // Global version query.
    uint64_t count = 4;
    uint64_t ids[4] = {};
    const char* names[4] = {};
    ffxQueryDescGetVersions versions{};
    versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    versions.outputCount = &count;
    versions.versionIds = ids;
    versions.versionNames = names;
    CHECK(api.query(nullptr, &versions.header) == FFX_API_RETURN_OK);
    CHECK(count == 1 && ids[0] == 0xFA4E && names[0] && std::strcmp(names[0], "fake AMD FSR 3.1") == 0);

    // An upscaler create without a D3D12 backend description cannot use WrongGPU: AMD's DLL gets it.
    ffxCreateContextDescUpscale up{};
    up.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    up.maxRenderSize = {kRW, kRH};
    up.maxUpscaleSize = {kOW, kOH};
    ffxContext upctx = nullptr;
    CHECK(api.create(&upctx, &up.header, nullptr) == FFX_API_RETURN_OK);
    int32_t phases = 0;
    ffxQueryDescUpscaleGetJitterPhaseCount pc{};
    pc.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT;
    pc.renderWidth = kRW;
    pc.displayWidth = kOW;
    pc.pOutPhaseCount = &phases;
    CHECK(api.query(&upctx, &pc.header) == FFX_API_RETURN_OK);
    CHECK(phases == 99);  // answered by the fake AMD DLL
    CHECK(api.destroy(&upctx, nullptr) == FFX_API_RETURN_OK);

    const std::string fake = FakeLog();
    std::printf("fake AMD DLL saw:\n%s", fake.c_str());
    CHECK(Count(fake, "create ") == 2 && Count(fake, "type=0x20001") == 1 && Count(fake, "type=0x10000") == 1);
    CHECK(Count(fake, "configure ") == 1 && Count(fake, "dispatch ") == 1 && Count(fake, "destroy ") == 2);
    CHECK(Count(fake, "type=0x4\n") == 1 && Count(fake, "type=0x10004") == 1);
}

void RunAmdMode(Api& api) {
    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    ffxCreateContextDescUpscale up{};
    up.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    up.header.pNext = &backend.header;
    up.maxRenderSize = {kRW, kRH};
    up.maxUpscaleSize = {kOW, kOH};
    ffxContext ctx = nullptr;
    CHECK(api.create(&ctx, &up.header, nullptr) == FFX_API_RETURN_OK);
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
    const std::string fake = FakeLog();
    CHECK(Count(fake, "type=0x10000") == 1 && Count(fake, "type=0x10001") == 1 && Count(fake, "destroy ") == 1);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Same-frame split (interface version 2) and the submission trace.
//
// The "game" records a small graphics pass around the FSR dispatch: before it, graphics root signature, a root CBV
// with the colour, root constants, render target, viewport, scissor, topology and pipeline; after it, only the
// pipeline again (as after AMD's FSR, which changes it) and a draw. The draw fills a marker texture with the colour
// (times the root constants on Windows), so it only lands when everything else survived the split. Under Wine, vkd3d
// keeps one set of root constants for compute and graphics (a compute dispatch overwrites the graphics ones even
// without any split), so there the shader leaves the root constants out.
// The FSR input is copied, inside the split list before the dispatch, from a texture that another list (B) fills and
// that the game submits ahead of the split list in the same ExecuteCommandLists call: the first half must run after B,
// or the upscale sees the previous frame's input. Mode split_control runs the same scenario without the split.

typedef HRESULT(WINAPI* PfnSerializeRoot)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
typedef HRESULT(WINAPI* PfnCompile)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR, LPCSTR, UINT, UINT,
                                    ID3DBlob**, ID3DBlob**);

const char kGamePassHlsl[] =
    "cbuffer C : register(b0) { float4 color; };\n"
    "cbuffer K : register(b1) { float4 k; };\n"
    "float4 VS(uint id : SV_VertexID) : SV_Position {\n"
    "    float2 p = float2((id << 1) & 2, id & 2);\n"
    "    return float4(p * float2(2, -2) + float2(-1, 1), 0, 1);\n"
    "}\n"
    "float4 PS() : SV_Target { return USE_K ? color * k : color; }\n";

bool UnderWine() {
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    return ntdll && GetProcAddress(ntdll, "wine_get_version") != nullptr;
}

constexpr UINT kMarker = 8;  // marker render target size

struct GamePass {
    ID3D12Resource* colors = nullptr;  // upload buffer: one 256-byte constant buffer per frame
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pso = nullptr;
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    ID3D12Resource* marker = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
};

bool MakeGamePass(Gpu& gpu, GamePass& g) {
    HMODULE d3d12 = GetModuleHandleW(L"d3d12.dll");
    HMODULE compiler = LoadLibraryW(L"d3dcompiler_47.dll");
    auto serialize = d3d12 ? reinterpret_cast<PfnSerializeRoot>(reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12SerializeRootSignature"))) : nullptr;
    auto compile = compiler ? reinterpret_cast<PfnCompile>(reinterpret_cast<void*>(GetProcAddress(compiler, "D3DCompile"))) : nullptr;
    if (!serialize || !compile) return false;
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[1].Constants.ShaderRegister = 1;
    params[1].Constants.Num32BitValues = 4;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rd{};
    rd.NumParameters = 2;
    rd.pParameters = params;
    ID3DBlob *blob = nullptr, *error = nullptr, *vs = nullptr, *ps = nullptr;
    if (FAILED(serialize(&rd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error)) ||
        FAILED(gpu.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&g.root))))
        return false;
    const D3D_SHADER_MACRO macros[] = {{"USE_K", UnderWine() ? "0" : "1"}, {nullptr, nullptr}};
    if (FAILED(compile(kGamePassHlsl, sizeof kGamePassHlsl - 1, "game", macros, nullptr, "VS", "vs_5_0", 0, 0, &vs, &error)) ||
        FAILED(compile(kGamePassHlsl, sizeof kGamePassHlsl - 1, "game", macros, nullptr, "PS", "ps_5_0", 0, 0, &ps, &error)))
        return false;
    g.colors = gpu.Buffer(4 * 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!g.colors) return false;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = g.root;
    pd.VS = {vs->GetBufferPointer(), vs->GetBufferSize()};
    pd.PS = {ps->GetBufferPointer(), ps->GetBufferSize()};
    pd.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pd.SampleMask = 0xFFFFFFFFu;
    pd.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pd.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pd.RasterizerState.DepthClipEnable = TRUE;
    pd.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pd.NumRenderTargets = 1;
    pd.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    pd.SampleDesc.Count = 1;
    if (FAILED(gpu.device->CreateGraphicsPipelineState(&pd, IID_PPV_ARGS(&g.pso)))) return false;
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = 1;
    if (FAILED(gpu.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&g.rtvHeap)))) return false;
    g.marker = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, kMarker, kMarker, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                           D3D12_RESOURCE_STATE_RENDER_TARGET);
    if (!g.marker) return false;
    g.rtv = g.rtvHeap->GetCPUDescriptorHandleForHeapStart();
    gpu.device->CreateRenderTargetView(g.marker, nullptr, g.rtv);
    return true;
}

// Before the FSR dispatch: the graphics state the game expects to find again after it. The marker gets rgba (slot
// selects the constant buffer, so frames in flight do not share one).
void GameStateBefore(ID3D12GraphicsCommandList* l, GamePass& g, const float rgba[4], int slot) {
    float* p = nullptr;
    g.colors->Map(0, nullptr, reinterpret_cast<void**>(&p));
    std::memcpy(reinterpret_cast<char*>(p) + slot * 256, rgba, 16);
    g.colors->Unmap(0, nullptr);
    const float one[4] = {1, 1, 1, 1};
    l->SetGraphicsRootSignature(g.root);
    l->SetGraphicsRootConstantBufferView(0, g.colors->GetGPUVirtualAddress() + UINT64(slot) * 256);
    l->SetGraphicsRoot32BitConstants(1, 4, one, 0);
    l->OMSetRenderTargets(1, &g.rtv, FALSE, nullptr);
    const D3D12_VIEWPORT vp{0, 0, float(kMarker), float(kMarker), 0, 1};
    const D3D12_RECT sc{0, 0, LONG(kMarker), LONG(kMarker)};
    l->RSSetViewports(1, &vp);
    l->RSSetScissorRects(1, &sc);
    l->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    l->SetPipelineState(g.pso);
}

// After it: the pipeline again (the upscale replaced it), then the draw.
void GameDrawAfter(ID3D12GraphicsCommandList* l, GamePass& g) {
    l->SetPipelineState(g.pso);
    l->DrawInstanced(3, 1, 0, 0);
}

bool CheckMarker(Gpu& gpu, GamePass& g, const float rgba[4], const char* label) {
    gpu.Begin();
    Gpu::Pending pending = gpu.CopyOut(g.marker, 16, D3D12_RESOURCE_STATE_RENDER_TARGET);
    CHECK(gpu.SubmitAndWait());
    std::vector<float> m = gpu.Read(pending);
    double worst = 0;
    for (UINT i = 0; i < kMarker * kMarker; ++i)
        for (int c = 0; c < 4; ++c) worst = std::max(worst, Abs(m[i * 4 + c], rgba[c]));
    std::printf("%s: game's draw after the upscale: worst marker difference %.3g (marker[0] = %g %g %g %g)\n", label, worst, m[0], m[1], m[2], m[3]);
    CHECK(worst < 1e-6);
    return worst < 1e-6;
}

// Textures of one split scenario: per-frame scenes, the staging texture list B fills, the FSR input and output.
struct SplitScene {
    Scene30 s;                        // colorTex (FSR input), depthTex, outTex; s.color is the current frame's render
    ID3D12Resource* scene[2] = {};    // the two frames' renders (COPY_SOURCE)
    std::vector<Float4> color[2];
    float jitter[2][2] = {{0.3125f, -0.2f}, {-0.1875f, 0.4f}};
    ID3D12Resource* stage = nullptr;  // filled by list B each frame (COPY_SOURCE when L reads it)
};

bool MakeSplitScene(Gpu& gpu, SplitScene& x) {
    if (!MakeScene30(gpu, x.s)) return false;
    gpu.Begin();
    for (int f = 0; f < 2; ++f) {
        x.color[f] = Render(x.jitter[f][0], x.jitter[f][1], 0, 0);
        x.scene[f] = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, kTexW, kTexH, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!x.scene[f]) return false;
        gpu.Upload(x.scene[f], x.color[f].data(), 16, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    std::vector<Float4> zero(kTexW * kTexH, Float4{0, 0, 0, 0});
    x.stage = gpu.Texture(DXGI_FORMAT_R32G32B32A32_FLOAT, kTexW, kTexH, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!x.stage) return false;
    gpu.Upload(x.stage, zero.data(), 16, D3D12_RESOURCE_STATE_COPY_DEST);
    // The FSR input starts as zeros too. MakeScene30 left it in NON_PIXEL_SHADER_RESOURCE, and Upload copies into a
    // texture in COPY_DEST (the D3D12 debug layer reports the mismatch; Wine does not check it).
    const D3D12_RESOURCE_BARRIER toCopy =
        Gpu::Transition(x.s.colorTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    gpu.list->ResourceBarrier(1, &toCopy);
    gpu.Upload(x.s.colorTex, zero.data(), 16, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return gpu.SubmitAndWait();
}

// List B of frame f: the scene into the staging texture, and the marker cleared.
void RecordListB(Gpu& gpu, SplitScene& x, GamePass& g, int f) {
    gpu.allocatorB->Reset();
    gpu.listB->Reset(gpu.allocatorB, nullptr);
    gpu.listB->CopyResource(x.stage, x.scene[f]);
    D3D12_RESOURCE_BARRIER b = Gpu::Transition(x.stage, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    gpu.listB->ResourceBarrier(1, &b);
    const float zero[4] = {0, 0, 0, 0};
    gpu.listB->ClearRenderTargetView(g.rtv, zero, 0, nullptr);
    gpu.listB->Close();
}

// The split list up to the FSR dispatch: the staging texture into the FSR input, then the game's graphics state.
void RecordBeforeDispatch(Gpu& gpu, SplitScene& x, GamePass& g, const float rgba[4], int slot) {
    D3D12_RESOURCE_BARRIER b[2] = {Gpu::Transition(x.s.colorTex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST)};
    gpu.list->ResourceBarrier(1, b);
    gpu.list->CopyResource(x.s.colorTex, x.stage);
    b[0] = Gpu::Transition(x.s.colorTex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b[1] = Gpu::Transition(x.stage, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    gpu.list->ResourceBarrier(2, b);
    GameStateBefore(gpu.list, g, rgba, slot);
}

// Records and submits one frame (B then the split list, in one ExecuteCommandLists call), waits, checks the upscale
// and the marker. dispatch() records the FSR call into gpu.list.
template <class Dispatch>
void RunSplitFrame(Gpu& gpu, SplitScene& x, GamePass& g, int frame, const char* label, Dispatch dispatch) {
    const int f = frame % 2;  // scene and jitter alternate
    const float rgba[4] = {0.25f + 0.5f * f, 0.5f, 0.75f, 0.5f + 0.125f * frame};
    RecordListB(gpu, x, g, f);
    gpu.Begin();
    RecordBeforeDispatch(gpu, x, g, rgba, frame % 4);
    dispatch(f);
    GameDrawAfter(gpu.list, g);
    CHECK(SUCCEEDED(gpu.list->Close()));
    ID3D12CommandList* lists[] = {gpu.listB, gpu.list};
    gpu.queue->ExecuteCommandLists(2, lists);
    CHECK(gpu.Wait());
    x.s.color = x.color[f];
    x.s.jx = x.jitter[f][0];
    x.s.jy = x.jitter[f][1];
    char text[64];
    std::snprintf(text, sizeof text, "%s frame %d", label, frame);
    CheckScene30(gpu, x.s, text);
    CheckMarker(gpu, g, rgba, text);
}

ffxDispatchDescUpscale ApiDispatchOf(Gpu& gpu, Scene30& s, float jx, float jy) {
    ffxDispatchDescUpscale d{};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    d.commandList = gpu.list;
    d.color = Resource(s.colorTex, FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.depth = Resource(s.depthTex, FFX_API_SURFACE_FORMAT_R32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
    d.output = Resource(s.outTex, FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT, kOutTexW, kOutTexH, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    d.jitterOffset = {jx, jy};
    d.motionVectorScale = {float(kRW), float(kRH)};
    d.renderSize = {kRW, kRH};
    d.upscaleSize = {kOW, kOH};
    d.preExposure = 1.0f;
    d.frameTimeDelta = 16.7f;
    return d;
}

ffxContext CreateApiContext(Api& api, Gpu& gpu) {
    ffxCreateBackendDX12Desc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = gpu.device;
    ffxCreateContextDescUpscale create{};
    create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    create.header.pNext = &backend.header;
    create.maxRenderSize = {kTexW, kTexH};
    create.maxUpscaleSize = {kOW, kOH};
    ffxContext ctx = nullptr;
    CHECK(api.create(&ctx, &create.header, nullptr) == FFX_API_RETURN_OK);
    return ctx;
}

// split_standin: FidelityFX API path, stand-in with split_test = 1; two frames, both split (the hooks are installed
// at create, before the game resets its list).
void RunSplitApi(Api& api, Gpu& gpu, const char* label) {
    SplitScene x;
    GamePass g;
    CHECK(MakeSplitScene(gpu, x));
    CHECK(MakeGamePass(gpu, g));
    if (g_failures) return;
    ffxContext ctx = CreateApiContext(api, gpu);
    for (int f = 0; f < 2; ++f)
        RunSplitFrame(gpu, x, g, f, label, [&](int k) {
            ffxDispatchDescUpscale d = ApiDispatchOf(gpu, x.s, x.jitter[k][0], x.jitter[k][1]);
            CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
        });
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
}

// split_fsr30: FSR 3.0 upscaler DLL with split_test = 1: the upscaler (and the hooks) are created at the first
// dispatch, after the game reset its list, so frame 0 is not split (WG_ERROR_NOT_READY, stand-in without split);
// frame 1 is.
void RunSplitFsr30(Gpu& gpu) {
    HMODULE m = LoadLibraryW((g_dir + L"\\ffx_fsr3upscaler_x64.dll").c_str());
    CHECK(m != nullptr);
    if (!m) return;
    auto create = Export<f30::ErrorCode (*)(void*, const f30::UpscalerContextDescription*)>(m, "ffxFsr3UpscalerContextCreate");
    auto dispatch = Export<f30::ErrorCode (*)(void*, const f30::UpscalerDispatchDescription*)>(m, "ffxFsr3UpscalerContextDispatch");
    auto destroy = Export<f30::ErrorCode (*)(void*)>(m, "ffxFsr3UpscalerContextDestroy");
    if (!create || !dispatch || !destroy) return;
    SplitScene x;
    GamePass g;
    CHECK(MakeSplitScene(gpu, x));
    CHECK(MakeGamePass(gpu, g));
    if (g_failures) return;
    std::vector<uint32_t> ctx(f30::kUpscalerContextBytes / 4);
    f30::UpscalerContextDescription cd{};
    cd.maxRenderSize = {kTexW, kTexH};
    cd.displaySize = {kOW, kOH};
    CHECK(create(ctx.data(), &cd) == f30::OK);
    for (int f = 0; f < 2; ++f)
        RunSplitFrame(gpu, x, g, f, "split_fsr30", [&](int k) {
            x.s.jx = x.jitter[k][0];
            x.s.jy = x.jitter[k][1];
            f30::UpscalerDispatchDescription d = UpDispatch30(gpu, x.s, false);
            CHECK(dispatch(ctx.data(), &d) == f30::OK);
        });
    CHECK(destroy(ctx.data()) == f30::OK);
}

// split_cyberpunk / hooks_trace_fsr3: the path Cyberpunk 2077 uses (results 1d6061b): the game calls ffx_fsr3 with an
// UPSCALING_ONLY context, which calls the upscaler DLL; WrongGPU's two DLLs share one set of hooks (the first to need
// them owns them). split_cyberpunk: stand-in with split_test, three frames (frame 0 not ready: the upscaler and the
// hooks are created at the first dispatch). hooks_trace_fsr3: upscaler = amd, d3d12_hooks = on; both DLLs note the
// same dispatch, the trace reports it once.
void RunFsr3Path(Gpu& gpu, bool trace) {
    SetEnvironmentVariableA("WG_FAKE_FSR3_CALLS_UPSCALER", "1");
    HMODULE m = LoadLibraryW((g_dir + L"\\ffx_fsr3_x64.dll").c_str());
    CHECK(m != nullptr);
    if (!m) return;
    auto create = Export<f30::ErrorCode (*)(void*, f30::Fsr3ContextDescription*)>(m, "ffxFsr3ContextCreate");
    auto dispatch = Export<f30::ErrorCode (*)(void*, const f30::Fsr3DispatchUpscaleDescription*)>(m, "ffxFsr3ContextDispatchUpscale");
    auto destroy = Export<f30::ErrorCode (*)(void*)>(m, "ffxFsr3ContextDestroy");
    if (!create || !dispatch || !destroy) return;
    SplitScene x;
    GamePass g;
    CHECK(MakeSplitScene(gpu, x));
    CHECK(MakeGamePass(gpu, g));
    if (g_failures) return;
    std::vector<uint32_t> ctx(f30::kFsr3ContextBytes / 4);
    f30::Fsr3ContextDescription cd{};
    cd.flags = f30::FSR3_FLAG_UPSCALING_ONLY;
    cd.maxRenderSize = {kTexW, kTexH};
    cd.upscaleOutputSize = {kOW, kOH};
    cd.displaySize = {kOW, kOH};
    CHECK(create(ctx.data(), &cd) == f30::OK);
    auto record = [&](int k) {
        f30::Fsr3DispatchUpscaleDescription d{};
        d.commandList = gpu.list;
        d.color = Res30(x.s.colorTex, f30::SF_R32G32B32A32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.depth = Res30(x.s.depthTex, f30::SF_R32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.upscaleOutput = Res30(x.s.outTex, f30::SF_R32G32B32A32_FLOAT, kOutTexW, kOutTexH, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = {x.jitter[k][0], x.jitter[k][1]};
        d.motionVectorScale = {float(kRW), float(kRH)};
        d.renderSize = {kRW, kRH};
        d.preExposure = 1.0f;
        CHECK(dispatch(ctx.data(), &d) == f30::OK);
    };
    if (trace) {
        // upscale = amd: the fake AMD upscaler writes nothing; only the trace is checked.
        for (int f = 0; f < 2; ++f) {
            gpu.Begin();
            record(f);
            CHECK(SUCCEEDED(gpu.list->Close()));
            ID3D12CommandList* l[] = {gpu.list};
            gpu.queue->ExecuteCommandLists(1, l);
            CHECK(gpu.Wait());
        }
    } else {
        for (int frame = 0; frame < 3; ++frame) RunSplitFrame(gpu, x, g, frame, "split_cyberpunk", record);
    }
    CHECK(destroy(ctx.data()) == f30::OK);
}

// split_release: a split list the game resets without submitting, and one it submits only after split_timeout_ms;
// both are released from the CPU and nothing hangs.
void RunSplitRelease(Api& api, Gpu& gpu) {
    SplitScene x;
    GamePass g;
    CHECK(MakeSplitScene(gpu, x));
    CHECK(MakeGamePass(gpu, g));
    if (g_failures) return;
    ffxContext ctx = CreateApiContext(api, gpu);
    const float rgba[4] = {0.1f, 0.2f, 0.3f, 1.0f};
    // 1) Split, then the game throws the list away (Close and Reset without ExecuteCommandLists). List B runs first, so
    //    the staging texture is in COPY_SOURCE when the released first half copies from it.
    RecordListB(gpu, x, g, 0);
    ID3D12CommandList* first[] = {gpu.listB};
    gpu.queue->ExecuteCommandLists(1, first);
    CHECK(gpu.Wait());
    gpu.Begin();
    RecordBeforeDispatch(gpu, x, g, rgba, 2);
    ffxDispatchDescUpscale d = ApiDispatchOf(gpu, x.s, x.jitter[0][0], x.jitter[0][1]);
    CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    CHECK(SUCCEEDED(gpu.list->Close()));
    CHECK(SUCCEEDED(gpu.list->Reset(gpu.allocator, nullptr)));  // released from the CPU here
    CHECK(SUCCEEDED(gpu.list->Close()));
    Sleep(200);  // the released first half runs on WrongGPU's queue; the game must not reset its allocator under it
    // 2) A normal frame afterwards works.
    RunSplitFrame(gpu, x, g, 1, "split_release", [&](int k) {
        ffxDispatchDescUpscale dk = ApiDispatchOf(gpu, x.s, x.jitter[k][0], x.jitter[k][1]);
        CHECK(api.dispatch(&ctx, &dk.header) == FFX_API_RETURN_OK);
    });
    // 3) Split, the list submitted only after the timeout (another submission in between notices it).
    RecordListB(gpu, x, g, 0);
    gpu.Begin();
    RecordBeforeDispatch(gpu, x, g, rgba, 3);
    d = ApiDispatchOf(gpu, x.s, x.jitter[0][0], x.jitter[0][1]);
    CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
    GameDrawAfter(gpu.list, g);
    CHECK(SUCCEEDED(gpu.list->Close()));
    ID3D12CommandList* b[] = {gpu.listB};
    gpu.queue->ExecuteCommandLists(1, b);  // B first, then the timeout passes
    CHECK(gpu.Wait());
    Sleep(400);
    gpu.allocatorB->Reset();               // an empty list: its submission lets the hook see the overdue split
    gpu.listB->Reset(gpu.allocatorB, nullptr);
    gpu.listB->Close();
    gpu.queue->ExecuteCommandLists(1, b);
    ID3D12CommandList* l[] = {gpu.list};
    gpu.queue->ExecuteCommandLists(1, l);
    CHECK(gpu.Wait());
    x.s.color = x.color[0];
    x.s.jx = x.jitter[0][0];
    x.s.jy = x.jitter[0][1];
    CheckScene30(gpu, x.s, "split_release late submission");
    CheckMarker(gpu, g, rgba, "split_release late submission");
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
}

// hooks_trace: upscaler = amd with d3d12_hooks = on: the hooks are installed at the first dispatch and log where the
// game submits the FSR list (nothing is split).
void RunHooksTrace(Api& api, Gpu& gpu) {
    Scene30 s;
    CHECK(MakeScene30(gpu, s));
    ffxContext ctx = CreateApiContext(api, gpu);
    for (int f = 0; f < 2; ++f) {
        gpu.allocatorB->Reset();
        gpu.listB->Reset(gpu.allocatorB, nullptr);
        gpu.listB->Close();
        gpu.Begin();
        ffxDispatchDescUpscale d = ApiDispatchOf(gpu, s, 0.25f, -0.25f);
        CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
        CHECK(SUCCEEDED(gpu.list->Close()));
        if (f == 0) {  // B and the FSR list in one call: position 2 of 2
            ID3D12CommandList* lists[] = {gpu.listB, gpu.list};
            gpu.queue->ExecuteCommandLists(2, lists);
        } else {       // B in its own call after the dispatch was recorded, then the FSR list
            ID3D12CommandList* b[] = {gpu.listB};
            ID3D12CommandList* l[] = {gpu.list};
            gpu.queue->ExecuteCommandLists(1, b);
            gpu.queue->ExecuteCommandLists(1, l);
        }
        CHECK(gpu.Wait());
    }
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
}

// network_*: upscaler = network with the test double of the runtime (tests/fake_runtime.cpp, wronggpu_runtime.dll
// in the test folder): loading from the FSR DLLs' folder, version negotiation, errors in words, the fall back to AMD.
// The fake writes nothing into the output; only the routing and the log are checked.
void RunNetwork(Api& api, Gpu& gpu, const std::string& mode) {
    Scene30 s;
    CHECK(MakeScene30(gpu, s));
    ffxContext ctx = CreateApiContext(api, gpu);
    ffxQueryGetProviderVersion pv{};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    CHECK(api.query(&ctx, &pv.header) == FFX_API_RETURN_OK);
    const bool statistics = mode.rfind("network_split_stats", 0) == 0;
    const bool nested = mode == "network_split_stats_nested";
    const bool twice = mode == "network_split_stats_twice";
    const bool own = mode == "network_dispatch_fail" || mode == "network_split" || statistics || mode == "network_v1";
    std::printf("%s: provider %s\n", mode.c_str(), pv.versionName ? pv.versionName : "(none)");
    // WrongGPU's own context answers with its model's label ("WrongGPU Model M"); AMD's fake answers none.
    const bool fromRuntime = pv.versionName && std::strstr(pv.versionName, "WrongGPU") != nullptr;
    CHECK(fromRuntime == own);
    ID3D12QueryHeap* query = nullptr;
    ID3D12Resource* result = nullptr;
    GamePass pass;
    if (statistics) {
        D3D12_QUERY_HEAP_DESC q{};
        q.Type = D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;
        q.Count = nested ? 2 : 1;
        CHECK(SUCCEEDED(gpu.device->CreateQueryHeap(&q, IID_PPV_ARGS(&query))));
        if (!query) return;
        gpu.keep.push_back(query);
        result = gpu.Buffer(2*sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        CHECK(result && MakeGamePass(gpu, pass));
        if (!result || !pass.pso) return;
    }
    for (int f = 0; f < 3; ++f) {
        gpu.Begin();
        if (statistics) {
            const float color[4] = {0.25f, 0.5f, 0.75f, 1};
            GameStateBefore(gpu.list, pass, color, f);
            gpu.list->BeginQuery(query, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
            GameDrawAfter(gpu.list, pass);
            if(nested) gpu.list->BeginQuery(query,D3D12_QUERY_TYPE_PIPELINE_STATISTICS,1);
        }
        ffxDispatchDescUpscale d = ApiDispatchOf(gpu, s, 0.25f, -0.25f);
        CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
        if (statistics) {
            GameDrawAfter(gpu.list, pass);
            if(twice) {
                CHECK(api.dispatch(&ctx,&d.header)==FFX_API_RETURN_OK);
                GameDrawAfter(gpu.list,pass);
            }
            if(nested) {
                gpu.list->EndQuery(query,D3D12_QUERY_TYPE_PIPELINE_STATISTICS,1);
                gpu.list->ResolveQueryData(query,D3D12_QUERY_TYPE_PIPELINE_STATISTICS,1,1,result,sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS));
            }
            gpu.list->EndQuery(query, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
            gpu.list->ResolveQueryData(query, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 1, result, 0);
        }
        CHECK(gpu.SubmitAndWait());
        if (statistics) {
            D3D12_QUERY_DATA_PIPELINE_STATISTICS* data = nullptr;
            D3D12_RANGE read{0, 2*sizeof(*data)};
            CHECK(SUCCEEDED(result->Map(0, &read, reinterpret_cast<void**>(&data))));
            if (data) {
                std::printf("statistics frame %d: IA vertices %llu, primitives %llu, CS invocations %llu\n", f,
                            data->IAVertices, data->IAPrimitives, data->CSInvocations);
                CHECK(data->IAVertices == (twice?9:6) && data->IAPrimitives == (twice?3:2) && data->CSInvocations == 0);
                if(nested) CHECK(data[1].IAVertices==3 && data[1].IAPrimitives==1 && data[1].CSInvocations==0);
                D3D12_RANGE written{0,0};
                result->Unmap(0, &written);
            }
        }
    }
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
    const std::string fake = FakeLog();
    const size_t amdDispatches = Count(fake, "dispatch ctx=");
    std::printf("%s: fake AMD DLL saw %zu dispatches\n", mode.c_str(), amdDispatches);
    if (mode == "network_dispatch_fail") CHECK(amdDispatches == 2);  // frames 1 and 2
    else if (own) CHECK(amdDispatches == 0);
    else CHECK(amdDispatches == 3);
}

// network_cyberpunk_fail: the FSR 3.0 path Cyberpunk 2077 uses (ffx_fsr3 -> upscaler DLL, UPSCALING_ONLY) with a runtime
// that fails from frame 1 on: frame 0 is the runtime's, frames 1 and 2 are AMD's upscaler's.
void RunNetworkFsr3(Gpu& gpu) {
    SetEnvironmentVariableA("WG_FAKE_FSR3_CALLS_UPSCALER", "1");
    HMODULE m = LoadLibraryW((g_dir + L"\\ffx_fsr3_x64.dll").c_str());
    CHECK(m != nullptr);
    if (!m) return;
    auto create = Export<f30::ErrorCode (*)(void*, f30::Fsr3ContextDescription*)>(m, "ffxFsr3ContextCreate");
    auto dispatch = Export<f30::ErrorCode (*)(void*, const f30::Fsr3DispatchUpscaleDescription*)>(m, "ffxFsr3ContextDispatchUpscale");
    auto destroy = Export<f30::ErrorCode (*)(void*)>(m, "ffxFsr3ContextDestroy");
    if (!create || !dispatch || !destroy) return;
    Scene30 s;
    CHECK(MakeScene30(gpu, s));
    std::vector<uint32_t> ctx(f30::kFsr3ContextBytes / 4);
    f30::Fsr3ContextDescription cd{};
    cd.flags = f30::FSR3_FLAG_UPSCALING_ONLY;
    cd.maxRenderSize = {kTexW, kTexH};
    cd.upscaleOutputSize = {kOW, kOH};
    cd.displaySize = {kOW, kOH};
    CHECK(create(ctx.data(), &cd) == f30::OK);
    for (int f = 0; f < 3; ++f) {
        gpu.Begin();
        f30::Fsr3DispatchUpscaleDescription d{};
        d.commandList = gpu.list;
        d.color = Res30(s.colorTex, f30::SF_R32G32B32A32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.depth = Res30(s.depthTex, f30::SF_R32_FLOAT, kTexW, kTexH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.upscaleOutput = Res30(s.outTex, f30::SF_R32G32B32A32_FLOAT, kOutTexW, kOutTexH, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = {s.jx, s.jy};
        d.motionVectorScale = {float(kRW), float(kRH)};
        d.renderSize = {kRW, kRH};
        d.preExposure = 1.0f;
        CHECK(dispatch(ctx.data(), &d) == f30::OK);
        CHECK(gpu.SubmitAndWait());
    }
    CHECK(destroy(ctx.data()) == f30::OK);
    const std::string up = FakeLogOf(L"ffx_fsr3upscaler_x64.wronggpu-original.dll");
    CHECK(Count(up, "upscaler.dispatch") == 2);
}

// network_park: parked upscalers (context_park_ms = 300) with the test double of the runtime in its split mode, and a
// frame whose render size does not fit its textures. The context is destroyed and created again with the same
// parameters: the second one takes the first one's upscaler over (the runtime is created once); the stand-in upscales
// the bad frame, cut to the texture, and the runtime the frames after it; the last upscaler is destroyed by the
// timer's thread.
void RunNetworkPark(Api& api, Gpu& gpu) {
    Scene30 s;
    CHECK(MakeScene30(gpu, s));
    ffxContext ctx = CreateApiContext(api, gpu);
    auto frame = [&](bool bad) {
        gpu.Begin();
        ffxDispatchDescUpscale d = ApiDispatchOf(gpu, s, 0.25f, -0.25f);
        if (bad) d.renderSize = {kTexW + 8, kTexH + 8};  // larger than the color texture
        CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
        CHECK(gpu.SubmitAndWait());
    };
    frame(false);
    frame(false);
    const ffxContext first = ctx;
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK && ctx == nullptr);
    ctx = CreateApiContext(api, gpu);
    CHECK(ctx != nullptr);
    std::printf("network_park: second context %p (first %p)\n", ctx, first);
    frame(false);
    frame(true);
    frame(false);
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
    Sleep(900);  // context_park_ms = 300: the timer's thread destroys the parked upscaler
    CHECK(Count(FakeLog(), "dispatch ctx=") == 0);
}

// park_real (run by hand, not a ctest; needs the real wronggpu_runtime.dll next to the test's DLLs and the model): what
// The Witcher 3 does at every step of its dynamic resolution, with the game's sizes (textures and output 2560x1440,
// render size 1706x960): destroy + create with the same parameters, timed on the calling thread; the output of the
// first three frames of every new context as a checksum (the same inputs for every context: a context that takes a
// parked upscaler over must give what a newly created one gives); the CPU time of a dispatch and the time of a frame
// in steady state; the upscaler's GPU memory. Environment: WG_TEST_PARK_MS (context_park_ms, default 3000; 0 = every
// create is a real one), WG_TEST_RUNTIME_OPTIONS (runtime_options), WG_TEST_PARK_CYCLES (default 12),
// WG_TEST_PARK_EXPOSURE (value, the default: no exposure input; auto: FFX_UPSCALE_ENABLE_AUTO_EXPOSURE; texture: a
// 1x1 exposure texture, as The Witcher 3 gives one).
uint16_t HalfOf(float v) {
    uint32_t b;
    std::memcpy(&b, &v, sizeof b);
    const int e = int((b >> 23) & 0xFF) - 127 + 15;
    if (v <= 0.0f || e <= 0) return 0;
    if (e >= 31) return 0x7BFF;
    return uint16_t((e << 10) | ((b >> 13) & 0x3FF));
}

void RunParkReal(Api& api, Gpu& gpu) {
    const UINT W = 2560, H = 1440, RW = 1706, RH = 960;
    char text[64] = "12";
    GetEnvironmentVariableA("WG_TEST_PARK_CYCLES", text, sizeof text);
    const int cycles = std::atoi(text);
    ID3D12Resource* color[2] = {};
    for (int k = 0; k < 2; ++k) {
        color[k] = gpu.Texture(DXGI_FORMAT_R16G16B16A16_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        std::vector<uint16_t> px(size_t(W) * H * 4);
        for (UINT y = 0; y < H; ++y)
            for (UINT x = 0; x < W; ++x) {
                const UINT sx = x + 7 * k, sy = y + 3 * k;
                const float check = ((sx / 16 + sy / 16) & 1) ? 0.6f : 0.15f;
                uint16_t* p = &px[(size_t(y) * W + x) * 4];
                p[0] = HalfOf(check + 0.3f * float(sx % 97) / 97.0f);
                p[1] = HalfOf(0.1f + 0.8f * float(sy % 193) / 193.0f);
                p[2] = HalfOf(0.05f + 1.5f * float((sx + sy) % 61) / 61.0f);
                p[3] = HalfOf(1.0f);
            }
        gpu.Begin();
        gpu.Upload(color[k], px.data(), 8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        CHECK(gpu.SubmitAndWait());
    }
    ID3D12Resource* depth = gpu.Texture(DXGI_FORMAT_R32_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* motion = gpu.Texture(DXGI_FORMAT_R16G16_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* out = gpu.Texture(DXGI_FORMAT_R16G16B16A16_FLOAT, W, H, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CHECK(color[0] && color[1] && depth && motion && out);
    if (!color[0] || !color[1] || !depth || !motion || !out) return;
    {
        std::vector<float> z(size_t(W) * H);
        for (UINT y = 0; y < H; ++y)
            for (UINT x = 0; x < W; ++x) z[size_t(y) * W + x] = 0.02f + 0.9f * float((x / 64 + y / 64) % 11) / 11.0f;
        std::vector<uint16_t> mv(size_t(W) * H * 2, 0);
        gpu.Begin();
        gpu.Upload(depth, z.data(), 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Upload(motion, mv.data(), 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        CHECK(gpu.SubmitAndWait());
    }
    char exposureMode[32] = "value";
    GetEnvironmentVariableA("WG_TEST_PARK_EXPOSURE", exposureMode, sizeof exposureMode);
    const bool autoExposure = std::strcmp(exposureMode, "auto") == 0;
    ID3D12Resource* exposure = nullptr;
    if (std::strcmp(exposureMode, "texture") == 0) {
        exposure = gpu.Texture(DXGI_FORMAT_R32_FLOAT, 1, 1, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        const float e = 0.35f;
        gpu.Begin();
        gpu.Upload(exposure, &e, 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        CHECK(exposure && gpu.SubmitAndWait());
    }
    std::printf("park_real: exposure: %s\n", exposure ? "texture" : autoExposure ? "auto" : "value");
    LARGE_INTEGER hz;
    QueryPerformanceFrequency(&hz);
    auto now = [] {
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        return t.QuadPart;
    };
    auto ms = [&](LONGLONG a, LONGLONG b) { return 1000.0 * double(b - a) / double(hz.QuadPart); };
    auto create = [&]() {
        ffxCreateBackendDX12Desc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backend.device = gpu.device;
        ffxCreateContextDescUpscale c{};
        c.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        c.header.pNext = &backend.header;
        c.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                  (autoExposure ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0);
        c.maxRenderSize = {W, H};
        c.maxUpscaleSize = {W, H};
        ffxContext ctx = nullptr;
        CHECK(api.create(&ctx, &c.header, nullptr) == FFX_API_RETURN_OK && ctx);
        return ctx;
    };
    static const float kJitter[4][2] = {{0.0f, -1.0f / 6}, {-0.25f, 1.0f / 6}, {0.25f, -7.0f / 18}, {-0.375f, -1.0f / 18}};
    ffxContext ctx = nullptr;
    double dispatchMs = 0;
    // One frame with input number i; hash: also the checksum of the output (read back after the frame).
    auto frame = [&](int i, uint64_t* hash) {
        gpu.Begin();
        ffxDispatchDescUpscale d{};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = gpu.list;
        d.color = Resource(color[i & 1], FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.depth = Resource(depth, FFX_API_SURFACE_FORMAT_R32_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.motionVectors = Resource(motion, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.output = Resource(out, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, W, H, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        if (exposure) d.exposure = Resource(exposure, FFX_API_SURFACE_FORMAT_R32_FLOAT, 1, 1, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.jitterOffset = {kJitter[i & 3][0], kJitter[i & 3][1]};
        d.motionVectorScale = {float(RW), float(RH)};
        d.renderSize = {RW, RH};
        d.upscaleSize = {W, H};
        d.preExposure = 1.0f;
        d.frameTimeDelta = 16.7f;
        d.cameraNear = 100000.0f;
        d.cameraFar = 0.1f;
        d.cameraFovAngleVertical = 1.0472f;
        const LONGLONG t0 = now();
        const ffxReturnCode_t rc = api.dispatch(&ctx, &d.header);
        dispatchMs = ms(t0, now());
        CHECK(rc == FFX_API_RETURN_OK);
        Gpu::Pending read;
        if (hash) read = gpu.CopyOut(out, 8, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        CHECK(gpu.SubmitAndWait());
        if (hash) {
            const std::vector<float> data = gpu.Read(read);
            uint64_t h = 14695981039346656037ull;
            const auto* bytes = reinterpret_cast<const unsigned char*>(data.data());
            for (size_t k = 0; k < data.size() * sizeof(float); ++k) h = (h ^ bytes[k]) * 1099511628211ull;
            *hash = h;
            read.buffer->Release();
            gpu.keep.erase(std::find(gpu.keep.begin(), gpu.keep.end(), static_cast<IUnknown*>(read.buffer)));
        }
    };
    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    };
    LONGLONG t0 = now();
    ctx = create();
    std::printf("park_real: first create %.1f ms\n", ms(t0, now()));
    if (!ctx) return;
    for (int i = 0; i < 30; ++i) frame(100 + i, nullptr);  // the lists are known, the upscaler has a history
    ffxQueryDescUpscaleGetGPUMemoryUsage memory{};
    FfxApiEffectMemoryUsage usage{};
    memory.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE;
    memory.gpuMemoryUsageUpscaler = &usage;
    if (api.query(&ctx, &memory.header) == FFX_API_RETURN_OK)
        std::printf("park_real: upscaler GPU memory %.0f MiB\n", double(usage.totalUsageInBytes) / 1048576.0);
    std::vector<double> destroys, creates, both, firsts;
    uint64_t reference[3] = {};
    bool same = true;
    for (int c = 0; c < cycles; ++c) {
        t0 = now();
        CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
        const LONGLONG t1 = now();
        ctx = create();
        const LONGLONG t2 = now();
        if (!ctx) return;
        uint64_t hash[3] = {};
        double first = 0;
        for (int f = 0; f < 3; ++f) {
            frame(f, &hash[f]);
            if (f == 0) first = dispatchMs;
        }
        for (int f = 0; f < 3; ++f) {
            if (c == 0) reference[f] = hash[f];
            same = same && hash[f] == reference[f];
        }
        std::printf("park_real: cycle %2d: destroy %7.3f ms, create %7.3f ms, first dispatch %6.3f ms; fnv1a64 %016llx %016llx %016llx\n", c,
                    ms(t0, t1), ms(t1, t2), first, static_cast<unsigned long long>(hash[0]), static_cast<unsigned long long>(hash[1]),
                    static_cast<unsigned long long>(hash[2]));
        destroys.push_back(ms(t0, t1));
        creates.push_back(ms(t1, t2));
        both.push_back(ms(t0, t2));
        firsts.push_back(first);
        for (int i = 0; i < 5 + c % 3; ++i) frame(200 + c + i, nullptr);  // the history differs from cycle to cycle
    }
    std::sort(both.begin(), both.end());
    std::printf("park_real: destroy + create over %d cycles: median %.3f ms, max %.3f ms (destroy median %.3f, create median %.3f); "
                "first dispatch median %.3f ms\n",
                cycles, median(both), both.empty() ? 0.0 : both.back(), median(destroys), median(creates), median(firsts));
    std::printf("park_real: the first three frames of every context are %s (fnv1a64 %016llx %016llx %016llx)\n",
                same ? "the same for all contexts" : "NOT the same for all contexts", static_cast<unsigned long long>(reference[0]),
                static_cast<unsigned long long>(reference[1]), static_cast<unsigned long long>(reference[2]));
    CHECK(same);
    std::vector<double> cpu, whole;
    for (int i = 0; i < 400; ++i) {
        t0 = now();
        frame(i, nullptr);
        if (i >= 100) cpu.push_back(dispatchMs), whole.push_back(ms(t0, now()));
    }
    double cpuSum = 0, wholeSum = 0;
    for (double v : cpu) cpuSum += v;
    for (double v : whole) wholeSum += v;
    std::printf("park_real: steady state over %zu frames: dispatch call median %.4f ms, mean %.4f ms; frame (record, submit, wait for the GPU) "
                "median %.3f ms, mean %.3f ms\n",
                cpu.size(), median(cpu), cpuSum / double(cpu.size()), median(whole), wholeSum / double(whole.size()));
    t0 = now();
    CHECK(api.destroy(&ctx, nullptr) == FFX_API_RETURN_OK);
    std::printf("park_real: last destroy %.3f ms\n", ms(t0, now()));
    char park[32] = "3000";
    GetEnvironmentVariableA("WG_TEST_PARK_MS", park, sizeof park);
    if (std::atoi(park) > 0) Sleep(DWORD(std::atoi(park)) + 1500);  // the timer's thread destroys the parked upscaler
}

// switch_real (run by hand, not a ctest; 0.5.1): a player switching upscalers in an upscaler mod's menu, as OptiScaler
// does it: the front end hosted (loaded as wronggpu_ffx_dx12.dll), the host lists the versions and creates each context
// with OVERRIDE_VERSION (WrongGPU Model M, WrongGPU Model K, AMD's version through the front end's AMD provider), the
// new context first, frames on it, then the old context destroyed. The folder holds the real wronggpu_runtime.dll and
// the AMD provider wronggpu-amd-compare.dll (the test double tests/fake_original.cpp, which takes `amd_mib` MiB of GPU
// memory per context as AMD's FSR 4 would). Each cycle: M, K, M, AMD, K, AMD (M <-> K, M -> AMD, AMD -> K, K -> AMD,
// AMD -> M). After every step: the process's GPU memory (DXGI local usage; HIP's own allocations count) and HIP's
// device memory in use (every process), right after the old context's destroy ("hot", a parked upscaler included) and
// once settle_ms later ("settled"). Arguments: switch_real <cycles> <frames per half step> <settle_ms> <amd_mib>.
struct ProcessGpuMemory {
    IUnknown* adapter = nullptr;  // IDXGIAdapter3
    typedef int (*PfnMemInfo)(size_t*, size_t*);
    bool Init(ID3D12Device* dev) {
        typedef HRESULT(WINAPI * PfnFactory)(REFIID, void**);
        HMODULE dxgi = LoadLibraryW(L"dxgi.dll");
        auto create = dxgi ? reinterpret_cast<PfnFactory>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1"))) : nullptr;
        IDXGIFactory4* f = nullptr;
        if (create && SUCCEEDED(create(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&f))) && f) {
            IDXGIAdapter1* a = nullptr;
            if (SUCCEEDED(f->EnumAdapterByLuid(dev->GetAdapterLuid(), __uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&a))) && a) {
                a->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter));
                a->Release();
            }
            f->Release();
        }
        return adapter != nullptr;
    }
    double ProcessMiB() const {  // this process, local segment
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (!adapter || FAILED(static_cast<IDXGIAdapter3*>(adapter)->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
            return -1;
        return double(info.CurrentUsage) / 1048576.0;
    }
    static double HipUsedMiB() {  // every process (HIP's device view); -1 before the runtime has loaded HIP
        HMODULE hip = GetModuleHandleA("amdhip64_7.dll");
        auto info = hip ? reinterpret_cast<PfnMemInfo>(reinterpret_cast<void*>(GetProcAddress(hip, "hipMemGetInfo"))) : nullptr;
        size_t freeB = 0, totalB = 0;
        if (!info || info(&freeB, &totalB) != 0) return -1;
        return double(totalB - freeB) / 1048576.0;
    }
};

void RunSwitchReal(Api& api, Gpu& gpu, int cycles, int frames, int settleMs) {
    const UINT W = 2560, H = 1440, RW = 1706, RH = 960;
    ID3D12Resource* color = gpu.Texture(DXGI_FORMAT_R16G16B16A16_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* depth = gpu.Texture(DXGI_FORMAT_R32_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* motion = gpu.Texture(DXGI_FORMAT_R16G16_FLOAT, W, H, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* out = gpu.Texture(DXGI_FORMAT_R16G16B16A16_FLOAT, W, H, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CHECK(color && depth && motion && out);
    if (!color || !depth || !motion || !out) return;
    {
        std::vector<uint16_t> px(size_t(W) * H * 4);
        for (UINT y = 0; y < H; ++y)
            for (UINT x = 0; x < W; ++x) {
                uint16_t* p = &px[(size_t(y) * W + x) * 4];
                p[0] = HalfOf(((x / 16 + y / 16) & 1) ? 0.6f : 0.15f);
                p[1] = HalfOf(0.1f + 0.8f * float(y % 193) / 193.0f);
                p[2] = HalfOf(0.05f + 1.5f * float((x + y) % 61) / 61.0f);
                p[3] = HalfOf(1.0f);
            }
        std::vector<float> z(size_t(W) * H);
        for (UINT y = 0; y < H; ++y)
            for (UINT x = 0; x < W; ++x) z[size_t(y) * W + x] = 0.02f + 0.9f * float((x / 64 + y / 64) % 11) / 11.0f;
        std::vector<uint16_t> mv(size_t(W) * H * 2, 0);
        gpu.Begin();
        gpu.Upload(color, px.data(), 8, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Upload(depth, z.data(), 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.Upload(motion, mv.data(), 4, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        CHECK(gpu.SubmitAndWait());
    }
    // The host's version list (QUERY_GET_VERSIONS): WrongGPU's Model M and Model K ids, and AMD's first version.
    uint64_t count = 0;
    ffxQueryDescGetVersions q{};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    q.device = gpu.device;
    q.outputCount = &count;
    CHECK(api.query(nullptr, &q.header) == FFX_API_RETURN_OK && count >= 3);
    std::vector<uint64_t> ids(size_t(count), 0);
    std::vector<const char*> names(size_t(count), nullptr);
    q.versionIds = ids.data();
    q.versionNames = names.data();
    CHECK(api.query(nullptr, &q.header) == FFX_API_RETURN_OK);
    uint64_t idM = 0, idK = 0, idAmd = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        std::printf("switch_real: version [%zu] 0x%016llx \"%s\"\n", i, static_cast<unsigned long long>(ids[i]), names[i] ? names[i] : "");
        if (names[i] && std::strstr(names[i], "WrongGPU Model M")) idM = ids[i];
        else if (names[i] && std::strstr(names[i], "WrongGPU Model K")) idK = ids[i];
        else if (!idAmd && !(names[i] && std::strstr(names[i], "WrongGPU"))) idAmd = ids[i];  // (not an RR entry)
    }
    CHECK(idM && idK && idAmd);
    if (!idM || !idK || !idAmd) return;
    auto create = [&](uint64_t version) {
        ffxCreateBackendDX12Desc backend{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
        backend.device = gpu.device;
        ffxOverrideVersion over{};
        over.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
        over.versionId = version;
        backend.header.pNext = &over.header;
        ffxCreateContextDescUpscale c{};
        c.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        c.header.pNext = &backend.header;
        c.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
        c.maxRenderSize = {W, H};
        c.maxUpscaleSize = {W, H};
        ffxContext ctx = nullptr;
        CHECK(api.create(&ctx, &c.header, nullptr) == FFX_API_RETURN_OK && ctx);
        return ctx;
    };
    static const float kJitter[4][2] = {{0.0f, -1.0f / 6}, {-0.25f, 1.0f / 6}, {0.25f, -7.0f / 18}, {-0.375f, -1.0f / 18}};
    int frameNo = 0;
    auto frame = [&](ffxContext& ctx) {
        gpu.Begin();
        ffxDispatchDescUpscale d{};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = gpu.list;
        d.color = Resource(color, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.depth = Resource(depth, FFX_API_SURFACE_FORMAT_R32_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.motionVectors = Resource(motion, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, W, H, FFX_API_RESOURCE_STATE_COMPUTE_READ);
        d.output = Resource(out, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, W, H, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = {kJitter[frameNo & 3][0], kJitter[frameNo & 3][1]};
        d.motionVectorScale = {float(RW), float(RH)};
        d.renderSize = {RW, RH};
        d.upscaleSize = {W, H};
        d.preExposure = 1.0f;
        d.frameTimeDelta = 16.7f;
        d.cameraNear = 100000.0f;
        d.cameraFar = 0.1f;
        d.cameraFovAngleVertical = 1.0472f;
        ++frameNo;
        CHECK(api.dispatch(&ctx, &d.header) == FFX_API_RETURN_OK);
        CHECK(gpu.SubmitAndWait());
    };
    ProcessGpuMemory pm;
    if (!pm.Init(gpu.device)) std::printf("switch_real: note: DXGI does not report the process's video memory\n");
    const double start = pm.ProcessMiB();
    std::printf("switch_real: start: process GPU memory %.0f MiB\n", start);
    struct Step {
        const char* name;
        uint64_t id;
    };
    const Step order[6] = {{"M", idM}, {"K", idK}, {"M", idM}, {"AMD", idAmd}, {"K", idK}, {"AMD", idAmd}};
    ffxContext current = nullptr;
    const char* currentName = "none";
    std::vector<double> settled;
    int step = 0;
    for (int c = 0; c < cycles; ++c) {
        for (const Step& s : order) {
            ffxContext next = create(s.id);
            if (!next) return;
            for (int f = 0; f < frames; ++f) frame(next);
            if (current) CHECK(api.destroy(&current, nullptr) == FFX_API_RETURN_OK && current == nullptr);
            for (int f = 0; f < frames; ++f) frame(next);
            const double hot = pm.ProcessMiB(), hotHip = ProcessGpuMemory::HipUsedMiB();
            Sleep(DWORD(settleMs));
            for (int f = 0; f < 2; ++f) frame(next);
            const double cool = pm.ProcessMiB(), coolHip = ProcessGpuMemory::HipUsedMiB();
            settled.push_back(cool);
            ++step;
            std::printf("switch_real: step %2d (cycle %d) %s -> %s: process GPU memory %.0f MiB hot, %.0f MiB settled; HIP device memory in "
                        "use (every process) %.0f / %.0f MiB\n",
                        step, c + 1, currentName, s.name, hot, cool, hotHip, coolHip);
            current = next;
            currentName = s.name;
        }
    }
    CHECK(api.destroy(&current, nullptr) == FFX_API_RETURN_OK);
    Sleep(DWORD(settleMs));
    const double end = pm.ProcessMiB();
    std::printf("switch_real: all contexts destroyed: process GPU memory %.0f MiB (start %.0f MiB), HIP device memory in use %.0f MiB\n", end,
                start, ProcessGpuMemory::HipUsedMiB());
    // Same step of each cycle (cycle 2 on: the first cycle loads what stays for the process, e.g. the HIP runtime).
    if (settled.size() >= 12) {
        const double first = settled[5], last = settled.back();
        std::printf("switch_real: after %d switches: %.2f GB (end of cycle 1: %.2f GB); growth per cycle after the first: %.1f MiB\n",
                    step - 1, last / 1024.0, first / 1024.0, (last - first) / double(cycles - 1));
    }
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "upscale";
    if (mode == "switch_real") g_dllName = L"wronggpu_ffx_dx12.dll";
    g_dir = ExeDirectory();
    const std::wstring ini = g_dir + L"\\wronggpu.ini", logPath = g_dir + L"\\wronggpu-fsr.log";
    DeleteFileW(logPath.c_str());
    DeleteFileW((logPath + L".old").c_str());
    std::string settings = "[wronggpu]\nupscaler = standin\n";
    if (mode == "lanczos") settings += "filter = lanczos\n";
    if (mode == "warpdiff" || mode == "motionsign") settings += "view = warpdiff\ndebug_gain = 1\n";
    if (mode == "motionsign") settings += "motion_sign_x = -1\nmotion_sign_y = -1\n";
    if (mode == "amd" || mode == "fsr30_log") settings = "upscaler = amd\n";
    if (mode == "hooks_trace" || mode == "hooks_trace_fsr3") settings = "upscaler = amd\nd3d12_hooks = on\n";
    if (mode.rfind("split_", 0) == 0 && mode != "split_control") settings += "split_test = 1\n";
    if (mode == "split_release") settings += "split_timeout_ms = 200\n";
    if (mode.rfind("network_", 0) == 0) {
        settings = "upscaler = network\n";
        if (mode == "network_missing") settings += "runtime_dll = missing_runtime.dll\n";
        if (mode == "network_split" || mode.rfind("network_split_stats",0)==0) settings += "runtime_options = timing_log=1;sync=pipelined ; the runtime's own keys\n";
        const char* fake = mode == "network_create_fail" ? "fail_create"
                         : mode == "network_dispatch_fail" || mode == "network_cyberpunk_fail" ? "fail_dispatch"
                         : mode == "network_split" || mode.rfind("network_split_stats",0)==0 || mode == "network_park" ? "split" : mode == "network_v1" ? "v1" : mode == "network_bad" ? "bad" : "ok";
        SetEnvironmentVariableA("WG_FAKE_RUNTIME", fake);
        if (mode == "network_park") settings += "context_park_ms = 300\n";
    }
    if (mode == "park_real") {
        char park[32] = "3000", options[2048] = "";
        GetEnvironmentVariableA("WG_TEST_PARK_MS", park, sizeof park);
        GetEnvironmentVariableA("WG_TEST_RUNTIME_OPTIONS", options, sizeof options);
        settings = std::string("upscaler = network\ncontext_park_ms = ") + park + "\n";
        if (options[0]) settings += std::string("runtime_options = ") + options + "\n";
    }
    if (mode == "switch_real") {
        // Arguments: cycles, frames per half step, settle ms, AMD MiB, then the wronggpu.ini lines (each argument one
        // line, e.g. "runtime_options = kernels_dir=..."). The test double of AMD's DLL reads its size from the process
        // environment; the D3D12 debug layer stays off (its memory would count).
        SetEnvironmentVariableA("WG_FAKE_AMD_MIB", argc > 5 ? argv[5] : "400");
        SetEnvironmentVariableA("WG_TEST_D3D12_DEBUG", "0");
        settings = "[fsr]\nupscaler = network\n";
        for (int i = 6; i < argc; ++i) settings += std::string(argv[i]) + "\n";
    }
    if (mode.rfind("view_", 0) == 0) settings += "view = " + mode.substr(5) + "\n";
    // The checks below count per-frame trace lines (dispatches, submits, split releases) of every frame: log them all
    // (the release default logs only the first frames of a context).
    if (mode != "park_real" && mode != "switch_real") settings += "log_dispatches = 1\n";
    WriteFileText(ini, settings);

    Api api;
    if (!api.Load()) {
        std::printf("FAIL: amd_fidelityfx_dx12.dll not loaded or exports missing (error %lu)\n", GetLastError());
        return 1;
    }
    if (mode == "forward") {
        RunForward(api);
    } else if (mode == "amd") {
        RunAmdMode(api);
    } else if (mode == "fsr30_log") {
        RunFsr30Upscaler(mode, nullptr);
    } else {
        Gpu gpu;
        if (!gpu.Init()) {
            std::printf("SKIP: no D3D12 device\n");
            DeleteFileW(ini.c_str());
            return 77;
        }
        if (mode == "upscale" || mode == "lanczos") RunUpscale(api, gpu, mode == "lanczos");
        else if (mode == "upscale_mips") { gpu.mips = 16; RunUpscale(api, gpu, false); }  // bug hunt F: the stand-in, mip 0, debug layer
        else if (mode == "warpdiff" || mode == "motionsign") RunWarp(api, gpu, mode == "motionsign");
        else if (mode == "view_motion" || mode == "view_depth" || mode == "view_jitter" || mode == "srgb") RunViewSmoke(api, gpu, mode);
        else if (mode == "fsr30_upscaler" || mode == "fsr30_upscaler_shared") RunFsr30Upscaler(mode, &gpu);
        else if (mode == "fsr30_fsr3_inner" || mode == "fsr30_fsr3_static" || mode == "fsr30_fsr3_upscaling_only") RunFsr30Fsr3(mode, gpu);
        else if (mode == "hooks_trace") RunHooksTrace(api, gpu);
        else if (mode == "split_standin" || mode == "split_control") RunSplitApi(api, gpu, mode.c_str());
        else if (mode == "split_fsr30") RunSplitFsr30(gpu);
        else if (mode == "split_release") RunSplitRelease(api, gpu);
        else if (mode == "split_cyberpunk" || mode == "hooks_trace_fsr3") RunFsr3Path(gpu, mode == "hooks_trace_fsr3");
        else if (mode == "network_cyberpunk_fail") RunNetworkFsr3(gpu);
        else if (mode == "network_park") RunNetworkPark(api, gpu);
        else if (mode == "park_real") RunParkReal(api, gpu);
        else if (mode == "switch_real")
            RunSwitchReal(api, gpu, argc > 2 ? std::atoi(argv[2]) : 6, argc > 3 ? std::atoi(argv[3]) : 20, argc > 4 ? std::atoi(argv[4]) : 3500);
        else if (mode.rfind("network_", 0) == 0) RunNetwork(api, gpu, mode);
        else {
            std::printf("unknown mode %s\n", mode.c_str());
            return 2;
        }
        gpu.CheckDebugMessages();
    }

    // Every FidelityFX call is in the log. The DLLs write their log from a background thread: wait until the file has
    // stopped growing (no change for 400 ms, at most 5 s) before reading it.
    {
        uint64_t last = ~0ull;
        int still = 0;
        for (int i = 0; i < 100 && still < 8; ++i) {
            WIN32_FILE_ATTRIBUTE_DATA a{};
            const uint64_t size = GetFileAttributesExW(logPath.c_str(), GetFileExInfoStandard, &a)
                                      ? (uint64_t(a.nFileSizeHigh) << 32) | a.nFileSizeLow
                                      : 0;
            still = size == last ? still + 1 : 0;
            last = size;
            Sleep(50);
        }
    }
    const std::string log = ReadFileText(logPath);
    const bool fsr30 = mode.rfind("fsr30_", 0) == 0;
    if (mode == "split_cyberpunk" || mode == "hooks_trace_fsr3" || mode == "network_cyberpunk_fail") {
        CHECK(Count(log, "[ffx_fsr3] #1 ffxFsr3ContextCreate from test_dll_d3d12.exe") == 1);
    } else if (mode == "split_fsr30") {
        CHECK(Count(log, "[ffx_fsr3upscaler] #1 ffxFsr3UpscalerContextCreate from test_dll_d3d12.exe") == 1);
    } else if (!fsr30) {
        CHECK(Count(log, " ffxCreateContext ") >= 1);
        CHECK(Count(log, " ffxDestroyContext ") >= 1);
    } else if (mode.find("fsr3_") != std::string::npos) {
        CHECK(Count(log, "[ffx_fsr3] #1 ffxFsr3ContextCreate from test_dll_d3d12.exe") == 1);
        CHECK(Count(log, "AMD's DLL is an FSR 3.0.x build") >= 1);
        CHECK(Count(log, mode == "fsr30_fsr3_static" ? "over upscaleOutput" : "handled in ffx_fsr3upscaler") == 1);
        if (mode == "fsr30_fsr3_upscaling_only") CHECK(Count(log, "AMD's upscaler skipped: no frame generation") == 1);
    } else {
        CHECK(Count(log, "[ffx_fsr3upscaler] #1 ffxFsr3UpscalerContextCreate from test_dll_d3d12.exe") == 1);
        CHECK(Count(log, "ffxFsr3UpscalerGetJitterPhaseCount from test_dll_d3d12.exe renderWidth=32 displayWidth=64 -> 77") == 1);
        CHECK(Count(log, "ffxFsr3UpscalerContextDestroy from test_dll_d3d12.exe") == 1);
    }
    const bool trace = mode.rfind("hooks_trace", 0) == 0;
    const bool toAmd = mode == "network_missing" || mode == "network_create_fail" || mode == "network_bad";  // no WrongGPU frame
    const bool parks = mode == "network_park" || mode == "park_real" || mode == "switch_real";  // several contexts
    if (mode != "forward" && mode != "amd" && mode != "fsr30_log" && !trace && !toAmd && !parks) CHECK(Count(log, "WrongGPU frame: frame=0 ") == 1);
    if (mode == "network_park") {
        CHECK(Count(log, "fake runtime created") == 1);  // the second context took the first one's upscaler over
        CHECK(Count(log, "its upscaler is parked for 300 ms (context_park_ms)") == 2);
        CHECK(Count(log, "took over the parked upscaler of U1 (same device, sizes, flags, options and model") == 1);
        CHECK(Count(log, "-> OK context=U2(WrongGPU network (fake runtime))") == 1);
        // U2's first frame starts the upscaler over (reset=2); the frame after the bad one only discards the history.
        CHECK(Count(log, "WrongGPU frame: frame=0 ") == 2 && Count(log, "reset=2") == 1 && Count(log, "reset=1") == 1);
        CHECK(Count(log, "note U2: frame 1: render size 48x38 exceeds the color texture 40x30: the stand-in upscales this frame (40x30)") == 1);
        CHECK(Count(log, "note U2: frame 2: WrongGPU upscales again after 1 frame(s)") == 1);
        CHECK(Count(log, "timer: parked upscaler of U2 destroyed") == 1);
        CHECK(Count(log, "refused") == 0 && Count(log, "ERROR:") == 0 && Count(log, " ERROR U") == 0);
        CHECK(Count(log, "for the rest of its life") == 0 && Count(log, "AMD FSR create") == 0);
    }
    if (mode == "park_real") {
        std::printf("park_real: log: %zu contexts took a parked upscaler over, %zu upscalers created, %zu frames of the stand-in, %zu errors\n",
                    Count(log, "took over the parked upscaler"), Count(log, "network runtime ready"), Count(log, "stand-in for this frame"),
                    Count(log, "ERROR"));
        for (const char* needle : {"network runtime ready", "HIP teardown", "parked upscaler of"}) {
            const size_t at = log.rfind(needle);
            if (at == std::string::npos) continue;
            const size_t a = log.rfind('\n', at), b = log.find('\n', at);
            std::printf("park_real: log: %s\n", log.substr(a + 1, b - a - 1).c_str());
        }
    }
    if (mode == "switch_real") {
        std::printf("switch_real: log: %zu upscalers created, %zu took a parked one over, %zu parked, %zu parked ones destroyed, %zu new "
                    "arenas, %zu arenas kept from an earlier context, %zu arenas released, %zu AMD contexts, %zu errors\n",
                    Count(log, "network runtime ready"), Count(log, "took over the parked upscaler"), Count(log, "its upscaler is parked"),
                    Count(log, "parked upscaler of U"), Count(log, "CreateSharedHandle S_OK"), Count(log, "kept from an earlier context"),
                    Count(log, "kept arena of"), Count(log, "the host chose AMD's upscaler version"), Count(log, "ERROR"));
    }
    // Same-frame split and submission trace (interface version 2).
    if (mode.rfind("split_", 0) != 0 && !trace && mode != "network_split" && mode.rfind("network_split_stats",0)!=0 && !parks) CHECK(Count(log, "d3d12 hooks:") == 0);  // auto: no hooks without a split backend
    // upscaler = network: the runtime from the FSR DLLs' folder, its errors in words, AMD after a failure.
    if (mode.rfind("network_", 0) == 0) {
        const bool loads = mode != "network_missing";
        if (mode == "network_park") CHECK(Count(log, "interface version 2, flags 0x1 (same-frame split)") == 1);
        CHECK(Count(log, "\\wronggpu_runtime.dll (runtime_dll = wronggpu_runtime.dll) loaded: file version") == (loads ? 1u : 0u));
    }
    if (mode == "network_missing") {
        // Once when loading (ERROR), once with the routing of the context.
        CHECK(Count(log, "missing_runtime.dll (runtime_dll = missing_runtime.dll) not found: it must be in WrongGPU's data folder (or, for older installs, next to WrongGPU's FSR DLLs)") == 2);
        CHECK(Count(log, "ERROR: WrongGPU runtime") == 1 && Count(log, "upscaler context goes to AMD FSR") == 1);
    }
    if (mode == "network_create_fail") {
        CHECK(Count(log, "interface version 2, flags 0x0") == 1);
        CHECK(Count(log, "upscaler ERROR: fake runtime: no HIP device found (hipGetDeviceCount returned 0)") == 1);
        CHECK(Count(log, "create failed: WG_ERROR_DEVICE (the D3D12 or HIP runtime returned an error)") == 1);
        CHECK(Count(log, "upscaler context goes to AMD FSR") == 1);
    }
    if (mode == "network_dispatch_fail") {
        CHECK(Count(log, "upscaler ERROR: fake runtime: hipModuleLaunchKernel failed (hipErrorLaunchFailure)") == 1);
        CHECK(Count(log, "\"WrongGPU network (fake runtime)\" dispatch failed: WG_ERROR_DEVICE (the D3D12 or HIP runtime returned an error)") == 1);
        CHECK(Count(log, "now runs on AMD FSR for the rest of its life") == 1);
    }
    if (mode == "network_split" || mode.rfind("network_split_stats",0)==0) {
        CHECK(Count(log, "interface version 2, flags 0x1 (same-frame split)") == 1);
        // runtime_options reach the runtime after the front end's keys.
        CHECK(Count(log, "fake runtime created (options: view=upscale;filter=bilinear;jitter_compensation=1;warp_weight=0.9;debug_gain=4;"
                         "split_test=0;timing_log=1;sync=pipelined)") == 1);
        CHECK(Count(log, "installing WrongGPU's D3D12 vtable hooks (a backend sets WG_BACKEND_NEEDS_SPLIT)") == 1);
        CHECK(Count(log, "first half on W1 behind order ticket 3,") == 1 && Count(log, "released in ExecuteCommandLists") == (mode=="network_split_stats_twice"?6:3));
        CHECK(Count(log, "refused") == 0 && Count(log, "ERROR:") == 0 && Count(log, " ERROR U") == 0);
    }
    if (mode == "network_v1") {
        CHECK(Count(log, "interface version 1, flags 0x0; asked for version 2: returned null") == 1);
        CHECK(Count(log, "create: interface=1 ") == 1);
    }
    if (mode == "network_bad")
        CHECK(Count(log, "offers no usable upscaler interface (asked for version 2: returned a table without name, create, dispatch or destroy; "
                         "asked for version 1: returned a table without name, create, dispatch or destroy)") == 2);
    if (mode == "network_cyberpunk_fail") {
        CHECK(Count(log, "-> FFX_OK (WrongGPU network (fake runtime); AMD's upscaler skipped: no frame generation)") == 1);
        CHECK(Count(log, "(AMD after WrongGPU network (fake runtime) failed; C1 stays on AMD)") == 1);
        CHECK(Count(log, "(AMD; WrongGPU network (fake runtime) failed earlier on C1)") == 1);
        CHECK(Count(log, "WrongGPU stand-in") == 0);
    }
    if (mode == "split_cyberpunk" || mode == "hooks_trace_fsr3") {
        // One owner of the hooks for both DLLs: ffx_fsr3 sees the dispatch first.
        CHECK(Count(log, "d3d12 hooks: installing") == 1 && Count(log, "[ffx_fsr3] d3d12 hooks: installing") == 1);
        CHECK(Count(log, "[ffx_fsr3upscaler] d3d12 hooks:") == 0);
        CHECK(Count(log, "submit: FSR list") == (mode == "split_cyberpunk" ? 3u : 2u));  // one per frame although both DLLs note it
    }
    if (mode == "split_cyberpunk") {
        CHECK(Count(log, "refused (WG_ERROR_NOT_READY)") == 1);
        CHECK(Count(log, "split S1 of list") >= 1 && Count(log, "split S2 of list") >= 1 && Count(log, "split S3 of list") == 0);
        CHECK(Count(log, "released in ExecuteCommandLists") == 2);
        CHECK(Count(log, "AMD's upscaler skipped: no frame generation") == 3);
    }
    if (mode == "hooks_trace") {
        CHECK(Count(log, "installing WrongGPU's D3D12 vtable hooks (d3d12_hooks = on)") == 1);
        CHECK(Count(log, "submit: FSR list") == 2);
        CHECK(Count(log, "position 2 of 2") == 1);
        CHECK(Count(log, "before this call: 0 lists in 0 calls") == 1);
        CHECK(Count(log, "before this call: 1 lists in 1 calls") == 1);
        CHECK(Count(log, "split S") == 0);
    }
    if (mode == "split_standin") {
        CHECK(Count(log, "installing WrongGPU's D3D12 vtable hooks (a backend sets WG_BACKEND_NEEDS_SPLIT)") == 1);
        CHECK(Count(log, "split S1 of list") == 1 && Count(log, "split S2 of list") == 1);
        CHECK(Count(log, "8 bindings restored") == 2);
        CHECK(Count(log, "released in ExecuteCommandLists") == 2);
        CHECK(Count(log, "refused") == 0);
        CHECK(Count(log, "split_test: first split done") == 1);
        CHECK(Count(log, "interface=2") == 1 && Count(log, "split=available") == 1);
    }
    if (mode == "split_fsr30") {
        CHECK(Count(log, "refused (WG_ERROR_NOT_READY)") == 1);
        CHECK(Count(log, "split_test: no split this frame") == 1);
        CHECK(Count(log, "split S1 of list") == 1 && Count(log, "split S2 of list") == 0);
        CHECK(Count(log, "released in ExecuteCommandLists") == 1);
    }
    if (mode == "split_release") {
        CHECK(Count(log, "the game reset the list without submitting it") == 1);
        CHECK(Count(log, "did not submit the list within split_timeout_ms") == 1);
        CHECK(Count(log, "order already released from the CPU") == 1);
        CHECK(Count(log, "first half on W1 behind order ticket 3") == 1);
        CHECK(Count(log, "was reset without being submitted") == 1);  // the trace forgets the discarded dispatch
    }
    if (g_failures) std::printf("---- wronggpu-fsr.log ----\n%s\n", log.c_str());
    DeleteFileW(ini.c_str());
    std::printf("%s: %s\n", mode.c_str(), g_failures ? "FAILED" : "passed");
    return g_failures ? 1 : 0;
}
