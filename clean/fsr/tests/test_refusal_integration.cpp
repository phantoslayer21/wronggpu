// SPDX-License-Identifier: MIT
// B8 (REVIEW-215d P2): the front end and the REAL network backend (the runtime DLL's Context, the same code a game runs) with
// a controllable executor (executor=dx12-test: the D3D12 route, scripted frame by frame), driving refused frames for model K
// and, as the control, model M; with and without AMD's FSR; repeated; with the stand-in working and failing; and with the
// real last resort (passthrough_d3d12.cpp) copying the game's colour into the output, checked on the output's pixels. HIP is
// never started: no amdhip64 module is loaded in this process and the backend never logs the HIP runtime's start.
//
// Windows, a D3D12 device (WARP is enough, no GPU slot needed; a GPU through gpu_slot also works) and the runtime DLL of a build
// (python clean/runner/build.py --rocm <ROCm> --out <build> --runtime-only). Not part of the CI (no runtime DLL there); the
// front end's fixtures are test_frontend.cpp's, included below. From the repository root, with the registry header generated
// (python clean/wgshared/tools/gen_registry.py --out-dir <gen>):
//   clang++ -std=c++17 -O1 -D_CRT_SECURE_NO_WARNINGS -I <gen>/include -I clean/wgshared/include -I clean/fsr/include
//     -I clean/third_party/fidelityfx-sdk-1.1.4 clean/fsr/tests/test_refusal_integration.cpp clean/fsr/src/passthrough_d3d12.cpp
//     clean/fsr/src/{frontend,param_map,ffx_describe,config,fsr30_hook,fsr30_map,amd_compare,async_create}.cpp
//     -ld3d12 -ldxgi -o test_refusal_integration.exe
//   test_refusal_integration.exe <build>\wronggpu_runtime.dll
#define main wg_frontend_unit_main
#include "test_frontend.cpp"
#undef main

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdlib>
#include <cstring>

#include "../src/passthrough_d3d12.h"

namespace {

template <class T>
void Release(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

struct Gpu {
    ID3D12Device* device = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    uint64_t fenceValue = 0;
    HANDLE event = nullptr;
    ID3D12Resource *color = nullptr, *depth = nullptr, *motion = nullptr, *output = nullptr;
};

constexpr UINT kRenderW = 1706, kRenderH = 960, kOutW = 2560, kOutH = 1440;

ID3D12Resource* MakeTexture(ID3D12Device* dev, DXGI_FORMAT fmt, UINT w, UINT h, D3D12_RESOURCE_FLAGS flags,
                            D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    ID3D12Resource* r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}

ID3D12Resource* MakeBuffer(ID3D12Device* dev, UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = type;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = d.DepthOrArraySize = d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}

void Wait(Gpu& g) {
    g.queue->Signal(g.fence, ++g.fenceValue);
    if (g.fence->GetCompletedValue() < g.fenceValue) {
        g.fence->SetEventOnCompletion(g.fenceValue, g.event);
        WaitForSingleObject(g.event, 60000);
    }
}

void Submit(Gpu& g) {
    g.list->Close();
    ID3D12CommandList* l[] = {g.list};
    g.queue->ExecuteCommandLists(1, l);
    Wait(g);
}

void Begin(Gpu& g) {
    g.alloc->Reset();
    g.list->Reset(g.alloc, nullptr);
}

// The colour texture's pixel (x, y): RGBA16F, a pattern the output's copy can be checked against
uint16_t Pixel(UINT x, UINT y, UINT channel) { return (uint16_t)(0x3400 + ((x * 7 + y * 13 + channel * 101) & 0x3FF)); }

bool MakeGpu(Gpu& g) {
    IDXGIFactory4* f = nullptr;
    IDXGIAdapter* warp = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f))) || FAILED(f->EnumWarpAdapter(IID_PPV_ARGS(&warp)))) return false;
    const HRESULT hr = D3D12CreateDevice(warp, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&g.device));
    warp->Release();
    f->Release();
    if (FAILED(hr)) return false;
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(g.device->CreateCommandQueue(&qd, IID_PPV_ARGS(&g.queue))) ||
        FAILED(g.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&g.alloc))) ||
        FAILED(g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.alloc, nullptr, IID_PPV_ARGS(&g.list))) ||
        FAILED(g.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&g.fence))))
        return false;
    g.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const D3D12_RESOURCE_STATES srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    g.color = MakeTexture(g.device, DXGI_FORMAT_R16G16B16A16_FLOAT, kRenderW, kRenderH, D3D12_RESOURCE_FLAG_NONE,
                          D3D12_RESOURCE_STATE_COPY_DEST);
    g.depth = MakeTexture(g.device, DXGI_FORMAT_R32_FLOAT, kRenderW, kRenderH, D3D12_RESOURCE_FLAG_NONE, srv);
    g.motion = MakeTexture(g.device, DXGI_FORMAT_R16G16_FLOAT, kRenderW, kRenderH, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | srv);
    g.output = MakeTexture(g.device, DXGI_FORMAT_R16G16B16A16_FLOAT, kOutW, kOutH,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!g.color || !g.depth || !g.motion || !g.output) return false;
    // the colour's pattern, through an upload buffer
    D3D12_RESOURCE_DESC cd = g.color->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT rows = 0;
    UINT64 rowBytes = 0, total = 0;
    g.device->GetCopyableFootprints(&cd, 0, 1, 0, &fp, &rows, &rowBytes, &total);
    ID3D12Resource* up = MakeBuffer(g.device, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!up) return false;
    uint8_t* map = nullptr;
    up->Map(0, nullptr, reinterpret_cast<void**>(&map));
    for (UINT y = 0; y < rows; ++y) {
        auto* row = reinterpret_cast<uint16_t*>(map + fp.Footprint.RowPitch * (size_t)y);
        for (UINT x = 0; x < kRenderW; ++x)
            for (UINT c = 0; c < 4; ++c) row[x * 4 + c] = Pixel(x, y, c);
    }
    up->Unmap(0, nullptr);
    Begin(g);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = g.color;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = up;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = fp;
    g.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = g.color;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    b.Transition.StateAfter = srv;
    g.list->ResourceBarrier(1, &b);
    Submit(g);
    up->Release();
    return true;
}

// Row 0 of the output texture, the first `pixels` pixels as RGBA16F halves
bool ReadOutputRow0(Gpu& g, UINT pixels, std::vector<uint16_t>* out) {
    D3D12_RESOURCE_DESC od = g.output->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp = {};
    UINT rows = 0;
    UINT64 rowBytes = 0, total = 0;
    g.device->GetCopyableFootprints(&od, 0, 1, 0, &fp, &rows, &rowBytes, &total);
    ID3D12Resource* rb = MakeBuffer(g.device, total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    if (!rb) return false;
    Begin(g);
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = g.output;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    g.list->ResourceBarrier(1, &b);
    D3D12_TEXTURE_COPY_LOCATION src = {}, dst = {};
    src.pResource = g.output;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = rb;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    g.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
    g.list->ResourceBarrier(1, &b);
    Submit(g);
    uint8_t* map = nullptr;
    rb->Map(0, nullptr, reinterpret_cast<void**>(&map));
    out->assign(reinterpret_cast<uint16_t*>(map), reinterpret_cast<uint16_t*>(map) + (size_t)pixels * 4);
    rb->Unmap(0, nullptr);
    rb->Release();
    return true;
}

void RealRefusals(const char* dllPath) {
    HMODULE dll = LoadLibraryA(dllPath);
    std::printf("runtime DLL %s\n", dll ? "loaded" : "NOT loaded");
    std::fflush(stdout);
    CHECK(dll != nullptr);
    if (!dll) return;
    auto get = reinterpret_cast<PfnWgGetUpscalerBackend>(reinterpret_cast<void*>(GetProcAddress(dll, "wgGetUpscalerBackend")));
    const WgUpscalerBackend* real = get ? get(WG_UPSCALER_INTERFACE_VERSION) : nullptr;
    CHECK(real != nullptr);
    if (!real) return;
    std::printf("backend %p\n", (const void*)real);
    std::fflush(stdout);
    Gpu g;
    CHECK(MakeGpu(g));
    std::printf("WARP device and textures: %s\n", g.output ? "ready" : "FAILED");
    std::fflush(stdout);
    if (!g.output) return;
    CHECK(GetModuleHandleA("amdhip64_7.dll") == nullptr && GetModuleHandleA("amdhip64.dll") == nullptr);

    struct Scenario {
        const char* name;
        bool amd, standInCreateFails, realPassThrough;
        const char* script;
        int frames;
    };
    // frame 0 is taken; the refused frames follow; 'o' after them is taken again (the context stays usable, no counter)
    const Scenario scenarios[] = {
        {"AMD available", true, false, false, "orrrrr", 6},
        {"no AMD, stand-in works, five refusals in a row then taken", false, false, false, "orrrrroo", 8},
        {"no AMD, stand-in cannot be made: the game's own colour is passed through", false, true, true, "orrr", 4},
    };
    for (const bool model_k : {true, false}) {   // model K, and model M as the control
        for (const Scenario& sc : scenarios) {
            std::printf("scenario: %s, %s\n", model_k ? "model K" : "model M", sc.name);
            std::fflush(stdout);
            const int failuresBefore = g_failures;
            Reset();
            g_originalAvailable = sc.amd;
            g_standInCreateFails = sc.standInCreateFails;
            g_passThroughOn = true;
            if (sc.realPassThrough) g_passThroughReal = [](const WgFrame& f, std::string* why) { return wgfsr::CopyColorToOutput(f, why); };
            g_useBackend = real;
            g_resources[g.color] = {dxgi::R16G16B16A16_FLOAT, kRenderW, kRenderH};
            g_resources[g.depth] = {dxgi::R32_FLOAT, kRenderW, kRenderH};
            g_resources[g.motion] = {dxgi::R16G16_FLOAT, kRenderW, kRenderH};
            g_resources[g.output] = {dxgi::R16G16B16A16_FLOAT, kOutW, kOutH};
            const std::string options = std::string("executor=dx12-test;test_script=") + sc.script + (model_k ? ";model=K" : "");
            _putenv_s("WG_RUNTIME_OPTIONS", options.c_str());
            Config config;
            config.hosted = true;
            auto fe = MakeFrontEnd(config);
            UpscaleCreate own(FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE, /*withOverride=*/true);
            own.version.versionId = model_k ? kWrongGpuKProviderVersionId : kWrongGpuProviderVersionId;
            own.backend.device = g.device;
            ffxContext ctx = nullptr;
            const std::string tag = std::string(model_k ? "model K, " : "model M, ") + sc.name;
            CHECK(fe->CreateContext(&ctx, &own.upscale.header, kMemCb) == FFX_API_RETURN_OK && fe->OwnContextCount() == 1);
            CHECK(CountLog("network runtime ready: executor dx12-scripted") == 1);
            ffxDispatchDescUpscale d = MakeDispatch();
            d.commandList = g.list;
            d.color = Res(g.color, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, kRenderW, kRenderH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            d.depth = Res(g.depth, FFX_API_SURFACE_FORMAT_R32_FLOAT, kRenderW, kRenderH, FFX_API_RESOURCE_STATE_COMPUTE_READ);
            d.motionVectors = Res(g.motion, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, kRenderW, kRenderH, FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
            d.output = Res(g.output, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, kOutW, kOutH, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
            d.renderSize = {kRenderW, kRenderH};
            d.upscaleSize = {kOutW, kOutH};
            Begin(g);
            std::vector<ffxReturnCode_t> rc;
            std::vector<UpscaleState> state;
            for (int i = 0; i < sc.frames; ++i) {
                rc.push_back(fe->Dispatch(&ctx, &d.header));
                state.push_back(g_statuses.empty() ? UpscaleState::Failed : g_statuses.back().state);
            }
            Submit(g);   // the game's list, with whatever the front end recorded in it, runs
            CHECK(g.device->GetDeviceRemovedReason() == S_OK);
            for (ffxReturnCode_t r : rc) CHECK(r == FFX_API_RETURN_OK);   // every frame has an upscaler, whatever the refusals
            CHECK(state[0] == UpscaleState::Network);                      // frame 0 is the real backend's
            // the refusals came from the real backend's code path (backend.cpp dispatch_record), one line, not a counter
            CHECK(CountLog(model_k ? "model K: the D3D12 executor does not take this frame"
                                   : "executor dx12 (default on RDNA 4) does not take this frame") == 1);
            CHECK(CountLog("ERROR_RUNTIME_ERROR") == 0);
            if (sc.amd) {
                // the first refused frame hands the context to AMD's FSR for good: the rest are AMD's
                CHECK(state[1] == UpscaleState::Amd && !g_amd.empty() && g_amd.back().fn == "dispatch");
                for (int i = 2; i < sc.frames; ++i) CHECK(state[i] == UpscaleState::Amd);
                CHECK(CountLog("WrongGPU frame: frame=") == 2 && g_standInFrames.empty());   // the backend was asked for frames 0 and 1 only
            } else if (!sc.standInCreateFails) {
                // the stand-in takes each refused frame (five in a row included); the context stays WrongGPU's: the taken
                // frames after the refusals are the real backend's again
                for (int i = 1; i <= 5; ++i) CHECK(state[i] == UpscaleState::StandIn);
                CHECK(state[6] == UpscaleState::Network && state[7] == UpscaleState::Network);
                CHECK(g_standInFrames.size() == 5 && g_amd.empty() && g_passFrames.empty());
                CHECK(CountLog("AMD FSR is not available for this context; the stand-in upscales the frames the upscaler does not take") == 1);
            } else {
                // no stand-in: the game's colour is copied into the output by the real last resort, checked on the pixels
                for (int i = 1; i < sc.frames; ++i)
                    CHECK(state[i] == UpscaleState::Failed && g_statuses.back().reason.find("passed through unscaled") != std::string::npos);
                CHECK(g_passFrames.size() == 3 && g_amd.empty());
                CHECK(CountLog("the game's own colour is passed through to the output unscaled") == 1);
                std::vector<uint16_t> px;
                CHECK(ReadOutputRow0(g, 8, &px) && px.size() == 32);
                bool same = px.size() == 32;
                for (UINT x = 0; same && x < 8; ++x)
                    for (UINT c = 0; c < 4; ++c) same = same && px[x * 4 + c] == Pixel(x, 0, c);
                CHECK(same);   // the output's first pixels are the colour's
            }
            // HIP never started: no HIP runtime line, no amdhip64 module
            CHECK(CountLog("about to start the HIP runtime") == 0);
            CHECK(GetModuleHandleA("amdhip64_7.dll") == nullptr && GetModuleHandleA("amdhip64.dll") == nullptr);
            CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
            if (g_failures != failuresBefore) {  // the log of a failed scenario
                std::lock_guard<std::mutex> lock(g_logMutex);
                for (const std::string& line : g_log) std::printf("    log: %s\n", line.c_str());
            }
            std::printf("  %s: states", tag.c_str());
            for (UpscaleState s : state) std::printf(" %d", (int)s);
            std::printf(", checks so far %d, failures %d\n", g_checks, g_failures);
            g_passThroughReal = nullptr;
        }
    }
    _putenv_s("WG_RUNTIME_OPTIONS", "");
    Release(g.color);
    Release(g.depth);
    Release(g.motion);
    Release(g.output);
}

}  // namespace

int main(int argc, char** argv) {
    const char* dll = argc > 1 ? argv[1] : std::getenv("WG_TEST_RUNTIME_DLL");
    if (!dll || !*dll) {
        std::printf("SKIP: no runtime DLL (argument or WG_TEST_RUNTIME_DLL)\n");
        return 0;
    }
    RealRefusals(dll);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
