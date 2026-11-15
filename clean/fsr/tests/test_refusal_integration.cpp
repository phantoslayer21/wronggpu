// SPDX-License-Identifier: MIT
// B8 (REVIEW-215d P2): the front end and the REAL network backend (the runtime DLL's Context, the same code a game runs) with a
// controllable executor (executor=dx12-test: the D3D12 route, scripted frame by frame), driving refused frames for model K and, as
// the control, model M; with and without AMD's FSR; repeated; with the stand-in working and with it failing. In every scenario
// the refusal comes from the backend's own code path (backend.cpp dispatch_record, refused_frame_action), never a counter, and
// HIP is never started: the runtime DLL's counter of HIP executors made stays 0 (wgDiagHipExecutorsMade, exported for this test
// only, not part of the public header), no amdhip64 module is loaded in this process, and the backend never logs the HIP runtime's
// start.
//
// Windows, a D3D12 device (WARP is enough, no GPU slot needed; a GPU through gpu_slot also works) and the runtime DLL of a build.
// It is a release check, not a CI test (CI has no runtime DLL): the release lead's script under tools/release builds it from this
// source and runs it with the DLL of the build; without the DLL it FAILS (exit 2), a gate that runs it cannot pass by skipping
// (--skip-if-missing is for a developer's loop). The front end's fixtures are test_frontend.cpp's, included below. Build, from
// the repository root, with the registry header generated (python clean/wgshared/tools/gen_registry.py --out-dir <gen>) and the
// runtime DLL built (python clean/runner/build.py --rocm <ROCm> --out <build> --runtime-only):
//   clang++ -std=c++17 -O1 -D_CRT_SECURE_NO_WARNINGS -I <gen>/include -I clean/wgshared/include -I clean/fsr/include
//     -I clean/third_party/fidelityfx-sdk-1.1.4 clean/fsr/tests/test_refusal_integration.cpp
//     clean/fsr/src/{frontend,param_map,ffx_describe,config,fsr30_hook,fsr30_map,amd_compare,async_create}.cpp
//     -ld3d12 -ldxgi -o test_refusal_integration.exe
//   test_refusal_integration.exe <build>/wronggpu_runtime.dll
#define main wg_frontend_unit_main
#include "test_frontend.cpp"
#undef main

#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdlib>
#include <cstring>

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
        FAILED(g.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, g.alloc, nullptr, IID_PPV_ARGS(&g.list))))
        return false;
    const D3D12_RESOURCE_STATES srv = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    g.color = MakeTexture(g.device, DXGI_FORMAT_R16G16B16A16_FLOAT, kRenderW, kRenderH, D3D12_RESOURCE_FLAG_NONE, srv);
    g.depth = MakeTexture(g.device, DXGI_FORMAT_R32_FLOAT, kRenderW, kRenderH, D3D12_RESOURCE_FLAG_NONE, srv);
    g.motion = MakeTexture(g.device, DXGI_FORMAT_R16G16_FLOAT, kRenderW, kRenderH, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | srv);
    g.output = MakeTexture(g.device, DXGI_FORMAT_R16G16B16A16_FLOAT, kOutW, kOutH, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return g.color && g.depth && g.motion && g.output;
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
    Gpu g;
    CHECK(MakeGpu(g));
    std::printf("WARP device and textures: %s\n", g.output ? "ready" : "FAILED");
    std::fflush(stdout);
    if (!g.output) return;
    using HipMadeFn = uint32_t (*)();
    const auto hipMade = reinterpret_cast<HipMadeFn>(reinterpret_cast<void*>(GetProcAddress(dll, "wgDiagHipExecutorsMade")));
    CHECK(hipMade != nullptr);   // the runtime DLL counts the HIP executors it makes (a diagnostic export)
    CHECK(GetModuleHandleA("amdhip64_7.dll") == nullptr && GetModuleHandleA("amdhip64.dll") == nullptr);

    struct Scenario {
        const char* name;
        bool amd, standInCreateFails;
        const char* script;
        int frames;
    };
    // frame 0 is taken; the refused frames follow; 'o' after them is taken again (the context stays usable, no counter)
    const Scenario scenarios[] = {
        {"AMD available", true, false, "orrrrr", 6},
        {"no AMD, stand-in works, five refusals in a row then taken", false, false, "orrrrroo", 8},
        {"no AMD, stand-in cannot be made: AMD and the stand-in both failed, the error goes back", false, true, "orrroo", 6},
    };
    for (const bool model_k : {true, false}) {   // model K, and model M as the control
        for (const Scenario& sc : scenarios) {
            std::printf("scenario: %s, %s\n", model_k ? "model K" : "model M", sc.name);
            std::fflush(stdout);
            const int failuresBefore = g_failures;
            Reset();
            g_originalAvailable = sc.amd;
            g_standInCreateFails = sc.standInCreateFails;
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
            g.alloc->Reset();
            g.list->Reset(g.alloc, nullptr);
            std::vector<ffxReturnCode_t> rc;
            std::vector<UpscaleState> state;
            for (int i = 0; i < sc.frames; ++i) {
                rc.push_back(fe->Dispatch(&ctx, &d.header));
                state.push_back(g_statuses.empty() ? UpscaleState::Failed : g_statuses.back().state);
            }
            g.list->Close();
            CHECK(rc[0] == FFX_API_RETURN_OK && state[0] == UpscaleState::Network);   // frame 0 is the real backend's
            // the refusals came from the real backend's code path (dispatch_record): K's and M's own line, once, not a counter
            CHECK(CountLog(model_k ? "model K: the D3D12 executor does not take this frame"
                                   : "executor dx12 (default on RDNA 4) does not take this frame") == 1);
            if (sc.amd) {
                // the first refused frame hands the context to AMD's FSR for good: the rest are AMD's
                for (int i = 0; i < sc.frames; ++i) CHECK(rc[i] == FFX_API_RETURN_OK);
                CHECK(state[1] == UpscaleState::Amd && !g_amd.empty() && g_amd.back().fn == "dispatch");
                for (int i = 2; i < sc.frames; ++i) CHECK(state[i] == UpscaleState::Amd);
                CHECK(CountLog("WrongGPU frame: frame=") == 2 && g_standInFrames.empty());   // the backend was asked for frames 0 and 1 only
            } else if (!sc.standInCreateFails) {
                // the stand-in takes each refused frame (five in a row included); the context stays WrongGPU's: the taken frames
                // after the refusals are the real backend's again
                for (int i = 0; i < sc.frames; ++i) CHECK(rc[i] == FFX_API_RETURN_OK);
                for (int i = 1; i <= 5; ++i) CHECK(state[i] == UpscaleState::StandIn);
                CHECK(state[6] == UpscaleState::Network && state[7] == UpscaleState::Network);
                CHECK(g_standInFrames.size() == 5 && g_amd.empty());
            } else {
                // neither AMD nor the stand-in: the refused frames (1-3) return the error with ONE line naming both failures; the
                // frames the backend takes afterwards (4, 5) are fine again
                for (int i = 1; i <= 3; ++i) CHECK(rc[i] != FFX_API_RETURN_OK && state[i] == UpscaleState::Failed);
                CHECK(rc[4] == FFX_API_RETURN_OK && rc[5] == FFX_API_RETURN_OK);
                CHECK(CountLog("no upscaler could take this frame: AMD FSR is not available; the stand-in could not be created") == 1);
                CHECK(g_amd.empty() && g_standInFrames.empty());
            }
            // HIP never started: no HIP runtime line, no amdhip64 module, no HIP executor constructed
            CHECK(CountLog("about to start the HIP runtime") == 0);
            CHECK(GetModuleHandleA("amdhip64_7.dll") == nullptr && GetModuleHandleA("amdhip64.dll") == nullptr);
            CHECK(hipMade && hipMade() == 0);
            if (g_failures != failuresBefore) {  // the log of a failed scenario
                std::lock_guard<std::mutex> lock(g_logMutex);
                for (const std::string& line : g_log) std::printf("    log: %s\n", line.c_str());
            }
            CHECK(fe->DestroyContext(&ctx, kMemCb) == FFX_API_RETURN_OK);
            std::printf("  %s: states", tag.c_str());
            for (UpscaleState s : state) std::printf(" %d", (int)s);
            std::printf(", checks so far %d, failures %d\n", g_checks, g_failures);
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
    const bool skipMissing = argc > 1 && std::strcmp(argv[1], "--skip-if-missing") == 0;
    const char* dll = argc > 1 && !skipMissing ? argv[1] : std::getenv("WG_TEST_RUNTIME_DLL");
    if (!dll || !*dll) {
        std::printf("%s: no runtime DLL (argument or WG_TEST_RUNTIME_DLL)\n", skipMissing ? "SKIP" : "FAIL");
        return skipMissing ? 0 : 2;
    }
    RealRefusals(dll);
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
