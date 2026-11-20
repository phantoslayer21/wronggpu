// SPDX-License-Identifier: MIT
// ffx_swapchain_test: a measurement experiment (test only, nothing shipped). It drives AMD's public FSR SDK frame
// generation swap chain with OUR stand-in frame as the generated frame, and reports what can be observed.
//
// It creates, through the documented AMD FSR API (loaded at run time from AMD's signed DLLs in a folder outside this
// repository; see fetch_ffx_sdk.py):
//   - the frame generation swap chain (ffxCreateContextDescFrameGenerationSwapChainForHwndDX12 + version struct),
//   - the frame generation context (ffxCreateContextDescFrameGeneration + DX12 backend + version struct),
// and configures frameGenerationCallback (ffxConfigureDescFrameGeneration) with our own callback, which copies a
// stand-in image into the generated-frame output. AMD's own interpolation is never dispatched.
//
// It reports:
//   a. return codes, callback calls and present counts: does the pairing work without AMD's interpolation;
//   b. GPU memory: the API's own memory queries and the process's video memory before/after context creation;
//   c. on-screen intervals from the swap chain's GetFrameStatistics (mean, sd, p99, misses), when it reports any;
//   d. observable signs of the pacing path: the runtime's own messages (global debug callback), the process's CPU
//      time, and the threads that used CPU during the run with their priorities. These are observations, not proof.
//
// Written for this project from AMD's public SDK documentation and headers (used at build time from the fetched
// folder, not copied here) and Microsoft's API documentation.
//
// usage: ffx_swapchain_test --sdk <folder with amd_fidelityfx_loader_dx12.dll> [--mode borderless|exclusive|window]
//        [--vsync 0|1] [--real-fps F | --real-fps-factor X] [--seconds S] [--no-fg] [--out file.csv]
//        ffx_swapchain_test --dwm-check     (read only: DWM's completed frames over 1 s; exit 0 if >= 50)
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#ifndef _WINDOWS
#define _WINDOWS
#endif
#include <windows.h>
#include <dcomp.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <tlhelp32.h>
#include <wrl/client.h>

#include <ffx_api.h>
#include <ffx_api_loader.h>
#include <ffx_api_types.h>
#include <dx12/ffx_api_dx12.h>
#include <ffx_framegeneration.h>
#include <dx12/ffx_api_framegeneration_dx12.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

struct Options {
    std::wstring sdk;
    std::string mode = "borderless";
    int vsync = 0;
    double real_fps = 0;             // 0: real_fps_factor x refresh rate
    double real_fps_factor = 0.45;   // 0.45: output (2 x real) at 0.9 x refresh, inside the VRR window
    double seconds = 5.0;
    bool no_fg = false;
    bool dwm_check = false;
    std::string out;
};

double g_qpc_ms = 0;
UINT64 g_t0 = 0;
double now_ms() {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (static_cast<double>(c.QuadPart) - static_cast<double>(g_t0)) * g_qpc_ms;
}
double qpc_to_ms(INT64 q) { return (static_cast<double>(q) - static_cast<double>(g_t0)) * g_qpc_ms; }

bool g_quit = false;
LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_CLOSE || (m == WM_KEYDOWN && w == VK_ESCAPE)) {
        g_quit = true;
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}
void pump() {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

// The runtime's own messages (ffxConfigureDescGlobalDebug), kept with their time.
std::vector<std::string> g_messages;
void ffx_message(uint32_t type, const wchar_t *message) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "%.3f ms %s: %ls", now_ms(), type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning",
                  message ? message : L"");
    g_messages.push_back(buf);
}

double dwm_frames_per_second() {
    COMPOSITION_FRAME_ID f0 = 0, f1 = 0;
    const double t0 = now_ms();
    if (FAILED(DCompositionGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &f0))) return -1;
    Sleep(1000);
    if (FAILED(DCompositionGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &f1))) return -1;
    return static_cast<double>(f1 - f0) * 1000.0 / (now_ms() - t0);
}

// Read-only display context; returns the refresh period in ms.
double print_display(HWND hwnd) {
    HMONITOR mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFOEXW mi = {};
    mi.cbSize = sizeof(mi);
    GetMonitorInfoW(mon, &mi);
    double period_ms = 1000.0 / 60.0;
    UINT32 np = 0, nm = 0;
    if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &np, &nm) == ERROR_SUCCESS) {
        std::vector<DISPLAYCONFIG_PATH_INFO> paths(np);
        std::vector<DISPLAYCONFIG_MODE_INFO> modes(nm);
        if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &np, paths.data(), &nm, modes.data(), nullptr) == ERROR_SUCCESS) {
            for (UINT32 i = 0; i < np; ++i) {
                DISPLAYCONFIG_SOURCE_DEVICE_NAME sn = {};
                sn.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
                sn.header.size = sizeof(sn);
                sn.header.adapterId = paths[i].sourceInfo.adapterId;
                sn.header.id = paths[i].sourceInfo.id;
                if (DisplayConfigGetDeviceInfo(&sn.header) != ERROR_SUCCESS || wcscmp(sn.viewGdiDeviceName, mi.szDevice))
                    continue;
                DISPLAYCONFIG_TARGET_DEVICE_NAME tn = {};
                tn.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_TARGET_NAME;
                tn.header.size = sizeof(tn);
                tn.header.adapterId = paths[i].targetInfo.adapterId;
                tn.header.id = paths[i].targetInfo.id;
                if (DisplayConfigGetDeviceInfo(&tn.header) == ERROR_SUCCESS)
                    std::printf("display: target \"%ls\", output technology %d, EDID ids valid %u\n",
                                tn.monitorFriendlyDeviceName, static_cast<int>(tn.outputTechnology),
                                tn.flags.edidIdsValid);
                const DISPLAYCONFIG_RATIONAL &rr = paths[i].targetInfo.refreshRate;
                if (rr.Numerator && rr.Denominator) {
                    period_ms = 1000.0 * rr.Denominator / rr.Numerator;
                    std::printf("display: refresh %.4f Hz, period %.4f ms\n",
                                static_cast<double>(rr.Numerator) / rr.Denominator, period_ms);
                }
            }
        }
    }
    return period_ms;
}

D3D12_RESOURCE_STATES to_d3d12(uint32_t s) {
    switch (s) {
    case FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    case FFX_API_RESOURCE_STATE_COMPUTE_READ: return D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case FFX_API_RESOURCE_STATE_PIXEL_READ: return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    case FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ:
        return D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    case FFX_API_RESOURCE_STATE_COPY_SRC: return D3D12_RESOURCE_STATE_COPY_SOURCE;
    case FFX_API_RESOURCE_STATE_COPY_DEST: return D3D12_RESOURCE_STATE_COPY_DEST;
    case FFX_API_RESOURCE_STATE_GENERIC_READ: return D3D12_RESOURCE_STATE_GENERIC_READ;
    case FFX_API_RESOURCE_STATE_PRESENT: return D3D12_RESOURCE_STATE_PRESENT;
    case FFX_API_RESOURCE_STATE_RENDER_TARGET: return D3D12_RESOURCE_STATE_RENDER_TARGET;
    default: return D3D12_RESOURCE_STATE_COMMON;
    }
}

void barrier(ID3D12GraphicsCommandList *cl, ID3D12Resource *r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    if (a == b) return;
    D3D12_RESOURCE_BARRIER br = {};
    br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    br.Transition.pResource = r;
    br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    br.Transition.StateBefore = a;
    br.Transition.StateAfter = b;
    cl->ResourceBarrier(1, &br);
}

// Our "generated frame": copy the stand-in image into every requested output.
struct CallbackState {
    ID3D12Resource *standin = nullptr;
    long calls = 0;
    long outputs_written = 0;
    long bad = 0;
    std::map<uint32_t, long> num_generated;
};
ffxReturnCode_t our_frame_generation(ffxDispatchDescFrameGeneration *p, void *user) {
    CallbackState *cs = static_cast<CallbackState *>(user);
    ++cs->calls;
    cs->num_generated[p ? p->numGeneratedFrames : 99]++;
    if (!p || !p->commandList) {
        ++cs->bad;
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    ID3D12GraphicsCommandList *cl = static_cast<ID3D12GraphicsCommandList *>(p->commandList);
    for (uint32_t i = 0; i < p->numGeneratedFrames && i < 4; ++i) {
        ID3D12Resource *dst = static_cast<ID3D12Resource *>(p->outputs[i].resource);
        if (!dst) {
            ++cs->bad;
            continue;
        }
        const D3D12_RESOURCE_STATES st = to_d3d12(p->outputs[i].state);
        barrier(cl, dst, st, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->CopyResource(dst, cs->standin);
        barrier(cl, dst, D3D12_RESOURCE_STATE_COPY_DEST, st);
        ++cs->outputs_written;
    }
    return FFX_API_RETURN_OK;
}

struct ThreadCpu {
    DWORD tid;
    ULONGLONG cpu100ns;
    int priority;
};
std::map<DWORD, ThreadCpu> thread_cpu() {
    std::map<DWORD, ThreadCpu> m;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap == INVALID_HANDLE_VALUE) return m;
    THREADENTRY32 te = {sizeof(te)};
    const DWORD pid = GetCurrentProcessId();
    for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
        if (te.th32OwnerProcessID != pid) continue;
        HANDLE th = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, te.th32ThreadID);
        if (!th) continue;
        FILETIME c, e, k, u;
        if (GetThreadTimes(th, &c, &e, &k, &u)) {
            const ULONGLONG t = (static_cast<ULONGLONG>(k.dwHighDateTime) << 32 | k.dwLowDateTime) +
                                (static_cast<ULONGLONG>(u.dwHighDateTime) << 32 | u.dwLowDateTime);
            m[te.th32ThreadID] = ThreadCpu{te.th32ThreadID, t, GetThreadPriority(th)};
        }
        CloseHandle(th);
    }
    CloseHandle(snap);
    return m;
}

UINT64 local_vram_usage(IDXGIAdapter3 *a) {
    DXGI_QUERY_VIDEO_MEMORY_INFO vi = {};
    if (a && SUCCEEDED(a->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vi))) return vi.CurrentUsage;
    return 0;
}

void print_versions(const ffxFunctions &fx, ID3D12Device *dev, uint64_t create_type, const char *what) {
    uint64_t count = 0;
    ffxQueryDescGetVersions q = {};
    q.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    q.createDescType = create_type;
    q.device = dev;
    q.outputCount = &count;
    ffxReturnCode_t rc = fx.Query(nullptr, &q.header);
    std::vector<uint64_t> ids(static_cast<size_t>(count));
    std::vector<const char *> names(static_cast<size_t>(count));
    q.versionIds = ids.data();
    q.versionNames = names.data();
    if (count) rc = fx.Query(nullptr, &q.header);
    std::printf("ffx: versions for %s: rc %u, count %llu:", what, rc, static_cast<unsigned long long>(count));
    for (uint64_t i = 0; i < count; ++i) std::printf(" [%s id 0x%llx]", names[i] ? names[i] : "?", ids[i]);
    std::printf("\n");
}

}  // namespace

int main(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--sdk") {
            const std::string s = next();
            o.sdk.assign(s.begin(), s.end());
        } else if (a == "--mode") o.mode = next();
        else if (a == "--vsync") o.vsync = std::atoi(next().c_str());
        else if (a == "--real-fps") o.real_fps = std::atof(next().c_str());
        else if (a == "--real-fps-factor") o.real_fps_factor = std::atof(next().c_str());
        else if (a == "--seconds") o.seconds = std::atof(next().c_str());
        else if (a == "--no-fg") o.no_fg = true;
        else if (a == "--dwm-check") o.dwm_check = true;
        else if (a == "--out") o.out = next();
        else {
            std::printf("usage: ffx_swapchain_test --sdk <dir> [--mode borderless|exclusive|window] [--vsync 0|1] "
                        "[--real-fps F] [--seconds S] [--no-fg] [--out file.csv] | --dwm-check\n");
            return 1;
        }
    }
    o.seconds = std::min(std::max(o.seconds, 1.0), 30.0);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    g_qpc_ms = 1000.0 / static_cast<double>(f.QuadPart);
    g_t0 = static_cast<UINT64>(c.QuadPart);

    if (o.dwm_check) {
        const double fps = dwm_frames_per_second();
        std::printf("dwm-check: completed composition frames per second %.1f (need >= 50)\n", fps);
        return fps >= 50.0 ? 0 : 3;
    }
    if (o.sdk.empty()) {
        std::printf("--sdk <folder with amd_fidelityfx_loader_dx12.dll> is required\n");
        return 1;
    }

    // ---- AMD's loader (signed DLL from the fetched SDK folder; the effect DLL is found next to it).
    const std::wstring loader = o.sdk + L"\\amd_fidelityfx_loader_dx12.dll";
    HMODULE mod = LoadLibraryExW(loader.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!mod) {
        std::printf("ffx: cannot load %ls (error %lu)\n", loader.c_str(), GetLastError());
        return 4;
    }
    ffxFunctions fx = {};
    ffxLoadFunctions(&fx, mod);
    if (!fx.CreateContext || !fx.Configure || !fx.Query || !fx.DestroyContext) {
        std::printf("ffx: loader exports missing\n");
        return 4;
    }
    {
        ffxConfigureDescGlobalDebug dbg = {};
        dbg.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG;
        dbg.fpMessage = ffx_message;
        dbg.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE;
        std::printf("ffx: global debug callback configure rc %u\n", fx.Configure(nullptr, &dbg.header));
    }

    // ---- window
    WNDCLASSW wc = {};
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"wg_ffx_swapchain_test";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    HMONITOR mon = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO mi = {sizeof(mi)};
    GetMonitorInfoW(mon, &mi);
    const bool windowed = o.mode == "window";
    RECT r = mi.rcMonitor;
    DWORD style = WS_POPUP;
    if (windowed) {
        style = WS_OVERLAPPEDWINDOW;
        r = RECT{mi.rcWork.left + 64, mi.rcWork.top + 64, mi.rcWork.left + 64 + 1280, mi.rcWork.top + 64 + 720};
        AdjustWindowRectEx(&r, style, FALSE, WS_EX_TOPMOST);
    }
    HWND hwnd = CreateWindowExW(WS_EX_TOPMOST, wc.lpszClassName, L"WrongGPU FFX swap chain test (Esc ends)", style,
                                r.left, r.top, r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance,
                                nullptr);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    RECT client;
    GetClientRect(hwnd, &client);
    const UINT w = client.right - client.left, h = client.bottom - client.top;
    const double period_ms = print_display(hwnd);
    std::printf("dwm: completed composition frames per second %.1f (before presenting)\n", dwm_frames_per_second());
    const double real_fps = o.real_fps > 0 ? o.real_fps : o.real_fps_factor * 1000.0 / period_ms;
    std::printf("run: mode %s, vsync %d, real frames %.2f/s (output %.2f/s with frame generation), %s, %.0f s\n",
                o.mode.c_str(), o.vsync, real_fps, o.no_fg ? real_fps : 2 * real_fps,
                o.no_fg ? "frame generation OFF" : "frame generation ON (our stand-in frame)", o.seconds);

    // ---- D3D12
    ComPtr<IDXGIFactory6> factory;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    ComPtr<IDXGIAdapter1> adapter;
    factory->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
    ComPtr<IDXGIAdapter3> adapter3;
    adapter.As(&adapter3);
    ComPtr<ID3D12Device> dev;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)))) {
        std::printf("D3D12CreateDevice failed\n");
        return 3;
    }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    BOOL tearing = FALSE;
    factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing));
    std::printf("dxgi: DXGI_FEATURE_PRESENT_ALLOW_TEARING = %d\n", tearing);

    print_versions(fx, dev.Get(), FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12,
                   "frame generation swap chain (DX12)");
    print_versions(fx, dev.Get(), FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION, "frame generation");

    // ---- the frame generation swap chain
    const DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;
    const UINT kBuffers = 3;
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = w;
    sd.Height = h;
    sd.Format = fmt;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = kBuffers;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT | (tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
    IDXGISwapChain4 *proxy_raw = nullptr;
    ffxCreateContextDescFrameGenerationSwapChainForHwndDX12 scd = {};
    scd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_FOR_HWND_DX12;
    scd.swapchain = &proxy_raw;
    scd.hwnd = hwnd;
    scd.desc = &sd;
    scd.fullscreenDesc = nullptr;
    scd.dxgiFactory = factory.Get();
    scd.gameQueue = queue.Get();
    ffxCreateContextDescFrameGenerationSwapChainVersionDX12 scv = {};
    scv.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_DX12;
    scv.version = FFX_FRAMEGENERATION_SWAPCHAIN_DX12_VERSION;
    scd.header.pNext = &scv.header;
    ffxContext sc_ctx = nullptr;
    ffxReturnCode_t rc = fx.CreateContext(&sc_ctx, &scd.header, nullptr);
    std::printf("ffx: create frame generation swap chain rc %u, swap chain %p\n", rc, static_cast<void *>(proxy_raw));
    if (rc != FFX_API_RETURN_OK || !proxy_raw) return 5;
    ComPtr<IDXGISwapChain4> sc;
    sc.Attach(proxy_raw);
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    sc->SetMaximumFrameLatency(2);
    HANDLE waitable = sc->GetFrameLatencyWaitableObject();
    if (o.mode == "exclusive") {
        const HRESULT fh = sc->SetFullscreenState(TRUE, nullptr);
        BOOL fs = FALSE;
        sc->GetFullscreenState(&fs, nullptr);
        std::printf("dxgi: SetFullscreenState(TRUE) hr 0x%08lx, GetFullscreenState %d\n", fh, fs);
    }

    // ---- the frame generation context (AMD's interpolation is created but never dispatched)
    const UINT64 vram0 = local_vram_usage(adapter3.Get());
    ffxCreateContextDescFrameGeneration fgd = {};
    fgd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    fgd.flags = 0;
    fgd.displaySize = FfxApiDimensions2D{w, h};
    fgd.maxRenderSize = FfxApiDimensions2D{w, h};
    fgd.backBufferFormat = ffxApiGetSurfaceFormatDX12(fmt);
    ffxCreateBackendDX12Desc backend = {};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backend.device = dev.Get();
    ffxCreateContextDescFrameGenerationVersion fgv = {};
    fgv.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
    fgv.version = FFX_FRAMEGENERATION_VERSION;
    fgd.header.pNext = &backend.header;
    backend.header.pNext = &fgv.header;
    ffxContext fg_ctx = nullptr;
    rc = fx.CreateContext(&fg_ctx, &fgd.header, nullptr);
    const UINT64 vram1 = local_vram_usage(adapter3.Get());
    std::printf("ffx: create frame generation context rc %u\n", rc);
    std::printf("memory: process local video memory %.1f MiB before, %.1f MiB after the frame generation context "
                "(+%.1f MiB)\n",
                vram0 / 1048576.0, vram1 / 1048576.0, (static_cast<double>(vram1) - static_cast<double>(vram0)) / 1048576.0);
    if (rc != FFX_API_RETURN_OK) return 6;
    {
        FfxApiEffectMemoryUsage mu = {};
        ffxQueryDescFrameGenerationGetGPUMemoryUsage q = {};
        q.header.type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATION_GPU_MEMORY_USAGE;
        q.gpuMemoryUsageFrameGeneration = &mu;
        const ffxReturnCode_t qrc = fx.Query(&fg_ctx, &q.header);
        std::printf("memory: ffxQueryDescFrameGenerationGetGPUMemoryUsage rc %u total %.1f MiB aliasable %.1f MiB\n",
                    qrc, mu.totalUsageInBytes / 1048576.0, mu.aliasableUsageInBytes / 1048576.0);
        FfxApiEffectMemoryUsage su = {};
        ffxQueryFrameGenerationSwapChainGetGPUMemoryUsageDX12 sq = {};
        sq.header.type = FFX_API_QUERY_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_GPU_MEMORY_USAGE_DX12;
        sq.gpuMemoryUsageFrameGenerationSwapchain = &su;
        const ffxReturnCode_t src = fx.Query(&sc_ctx, &sq.header);
        std::printf("memory: swap chain GPU memory query rc %u total %.1f MiB aliasable %.1f MiB\n", src,
                    su.totalUsageInBytes / 1048576.0, su.aliasableUsageInBytes / 1048576.0);
    }

    // ---- our resources: RTVs on the swap chain buffers, the stand-in image
    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = kBuffers + 1;
    ComPtr<ID3D12DescriptorHeap> heap;
    dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    auto rtv = [&](UINT i) {
        D3D12_CPU_DESCRIPTOR_HANDLE hnd = heap->GetCPUDescriptorHandleForHeapStart();
        hnd.ptr += static_cast<SIZE_T>(i) * inc;
        return hnd;
    };
    std::vector<ComPtr<ID3D12Resource>> bufs(kBuffers);
    for (UINT i = 0; i < kBuffers; ++i) {
        sc->GetBuffer(i, IID_PPV_ARGS(&bufs[i]));
        dev->CreateRenderTargetView(bufs[i].Get(), nullptr, rtv(i));
    }
    ComPtr<ID3D12Resource> standin;
    {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC rd = bufs[0]->GetDesc();
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                     IID_PPV_ARGS(&standin));
        dev->CreateRenderTargetView(standin.Get(), nullptr, rtv(kBuffers));
    }
    std::vector<ComPtr<ID3D12CommandAllocator>> alloc(kBuffers);
    for (auto &a : alloc) dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a));
    ComPtr<ID3D12GraphicsCommandList> cl;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[0].Get(), nullptr, IID_PPV_ARGS(&cl));
    const float real_grey[4] = {0.20f, 0.20f, 0.22f, 1.0f};
    const float standin_grey[4] = {0.24f, 0.24f, 0.26f, 1.0f};
    cl->ClearRenderTargetView(rtv(kBuffers), standin_grey, 0, nullptr);
    barrier(cl.Get(), standin.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->Close();
    ID3D12CommandList *first[] = {cl.Get()};
    queue->ExecuteCommandLists(1, first);
    ComPtr<ID3D12Fence> fence;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE fev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 next_fence = 1;
    queue->Signal(fence.Get(), next_fence);
    fence->SetEventOnCompletion(next_fence++, fev);
    WaitForSingleObject(fev, 5000);
    std::vector<UINT64> fval(kBuffers, 0);

    CallbackState cbs;
    cbs.standin = standin.Get();
    ffxConfigureDescFrameGeneration cfg = {};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain = sc.Get();
    cfg.presentCallback = nullptr;
    cfg.frameGenerationCallback = our_frame_generation;
    cfg.frameGenerationCallbackUserContext = &cbs;
    cfg.frameGenerationEnabled = !o.no_fg;
    cfg.allowAsyncWorkloads = false;
    cfg.flags = 0;
    cfg.onlyPresentGenerated = false;
    cfg.generationRect = FfxApiRect2D{0, 0, static_cast<int32_t>(w), static_cast<int32_t>(h)};

    // ---- run
    struct Shown {
        double ms;
        UINT refresh;
    };
    std::map<UINT, Shown> shown;  // swap chain PresentCount -> on-screen time
    std::map<ffxReturnCode_t, long> cfg_rc;
    long stats_fail = 0, stats_ok = 0;
    HRESULT last_present_hr = S_OK;
    long present_fail = 0;
    auto poll_stats = [&]() {
        DXGI_FRAME_STATISTICS fs = {};
        if (FAILED(sc->GetFrameStatistics(&fs))) {
            ++stats_fail;
            return;
        }
        ++stats_ok;
        if (fs.PresentCount == 0 || fs.SyncQPCTime.QuadPart == 0 || shown.count(fs.PresentCount)) return;
        const double t = qpc_to_ms(fs.SyncQPCTime.QuadPart) -
                         static_cast<double>(fs.SyncRefreshCount - fs.PresentRefreshCount) * period_ms;
        shown[fs.PresentCount] = Shown{t, fs.PresentRefreshCount};
    };
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    const double real_period = 1000.0 / real_fps;
    const int frames = static_cast<int>(o.seconds * real_fps);
    FILETIME pc0, pe0, pk0, pu0, pc1, pe1, pk1, pu1;
    GetProcessTimes(GetCurrentProcess(), &pc0, &pe0, &pk0, &pu0);
    const std::map<DWORD, ThreadCpu> th0 = thread_cpu();
    const double run_t0 = now_ms();
    double next_t = run_t0;
    UINT first_count = 0, last_count = 0;
    sc->GetLastPresentCount(&first_count);
    for (int i = 0; i < frames && !g_quit; ++i) {
        pump();
        // Frame limiter: wait for the next real frame time, polling the swap chain's statistics every ~1 ms.
        next_t += real_period;
        while (now_ms() < next_t) {
            poll_stats();
            LARGE_INTEGER due;
            due.QuadPart = -10000;  // 1 ms
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, 5);
        }
        WaitForSingleObjectEx(waitable, 1000, TRUE);
        const UINT b = sc->GetCurrentBackBufferIndex();
        if (fence->GetCompletedValue() < fval[b]) {
            fence->SetEventOnCompletion(fval[b], fev);
            WaitForSingleObject(fev, 1000);
        }
        alloc[b]->Reset();
        cl->Reset(alloc[b].Get(), nullptr);
        barrier(cl.Get(), bufs[b].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cl->ClearRenderTargetView(rtv(b), real_grey, 0, nullptr);
        barrier(cl.Get(), bufs[b].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
        cl->Close();
        ID3D12CommandList *lists[] = {cl.Get()};
        queue->ExecuteCommandLists(1, lists);
        cfg.frameID = static_cast<uint64_t>(i) + 1;
        cfg_rc[fx.Configure(&fg_ctx, &cfg.header)]++;
        const UINT flags = (o.vsync == 0 && tearing && o.mode != "exclusive") ? DXGI_PRESENT_ALLOW_TEARING : 0;
        const HRESULT phr = sc->Present(o.vsync ? 1 : 0, flags);
        if (FAILED(phr)) {
            ++present_fail;
            last_present_hr = phr;
        }
        fval[b] = next_fence;
        queue->Signal(fence.Get(), next_fence++);
        poll_stats();
    }
    for (int k = 0; k < 100; ++k) {  // let the last presents reach the screen
        poll_stats();
        Sleep(1);
    }
    const double run_ms = now_ms() - run_t0;
    sc->GetLastPresentCount(&last_count);
    GetProcessTimes(GetCurrentProcess(), &pc1, &pe1, &pk1, &pu1);
    const std::map<DWORD, ThreadCpu> th1 = thread_cpu();

    // ---- shutdown, in AMD's documented order
    cfg.frameGenerationEnabled = false;
    cfg.frameGenerationCallback = nullptr;
    cfg.frameGenerationCallbackUserContext = nullptr;
    cfg.presentCallback = nullptr;
    const ffxReturnCode_t dis_rc = fx.Configure(&fg_ctx, &cfg.header);
    const ffxReturnCode_t fgd_rc = fx.DestroyContext(&fg_ctx, nullptr);
    if (o.mode == "exclusive") sc->SetFullscreenState(FALSE, nullptr);
    const ffxReturnCode_t scd_rc = fx.DestroyContext(&sc_ctx, nullptr);
    queue->Signal(fence.Get(), next_fence);
    fence->SetEventOnCompletion(next_fence, fev);
    WaitForSingleObject(fev, 2000);
    bufs.clear();
    const ULONG refs = sc.Reset();
    DestroyWindow(hwnd);

    // ---- report
    std::printf("result a: frames %d in %.1f ms; configure rc:", frames, run_ms);
    for (auto &kv : cfg_rc) std::printf(" %u x%ld;", kv.first, kv.second);
    std::printf(" Present failures %ld (last hr 0x%08lx); callback calls %ld, outputs written %ld, bad %ld; "
                "numGeneratedFrames:",
                present_fail, last_present_hr, cbs.calls, cbs.outputs_written, cbs.bad);
    for (auto &kv : cbs.num_generated) std::printf(" %u x%ld;", kv.first, kv.second);
    std::printf(" swap chain present count %u -> %u (+%u for %d app presents)\n", first_count, last_count,
                last_count - first_count, frames);
    std::printf("result a: shutdown: disable configure rc %u, destroy frame generation context rc %u, destroy swap "
                "chain context rc %u, swap chain refs after release %lu\n",
                dis_rc, fgd_rc, scd_rc, refs);
    auto ft = [](const FILETIME &x) { return (static_cast<ULONGLONG>(x.dwHighDateTime) << 32) | x.dwLowDateTime; };
    const double cpu_ms = ((ft(pk1) + ft(pu1)) - (ft(pk0) + ft(pu0))) / 10000.0;
    std::printf("result d: process CPU time during the run %.1f ms over %.1f ms wall (%.1f %% of one core)\n", cpu_ms,
                run_ms, 100.0 * cpu_ms / run_ms);
    std::vector<std::pair<double, ThreadCpu>> busy;
    for (auto &kv : th1) {
        auto it = th0.find(kv.first);
        const ULONGLONG before = it == th0.end() ? 0 : it->second.cpu100ns;
        busy.push_back({(kv.second.cpu100ns - before) / 10000.0, kv.second});
    }
    std::sort(busy.begin(), busy.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    std::printf("result d: threads in the process at the end %zu (at the start of the run %zu); busiest:", th1.size(),
                th0.size());
    for (size_t k = 0; k < busy.size() && k < 5; ++k)
        std::printf(" [tid %lu %.1f ms (%.1f %%) priority %d%s]", busy[k].second.tid, busy[k].first,
                    100.0 * busy[k].first / run_ms, busy[k].second.priority,
                    th0.count(busy[k].second.tid) ? "" : " new");
    std::printf("\n");
    std::printf("result d: runtime messages %zu\n", g_messages.size());
    for (size_t k = 0; k < g_messages.size() && k < 60; ++k) std::printf("  ffx message: %s\n", g_messages[k].c_str());

    std::printf("result c: GetFrameStatistics ok %ld, failed %ld, distinct presents with an on-screen time %zu\n",
                stats_ok, stats_fail, shown.size());
    const double target = (o.no_fg ? real_period : real_period / 2.0);
    std::vector<double> iv;
    int misses = 0;
    UINT prev = 0;
    double prev_ms = 0;
    int skipped_start = 0;
    for (auto &kv : shown) {
        if (skipped_start < 10) {  // start-up
            ++skipped_start;
            prev = kv.first;
            prev_ms = kv.second.ms;
            continue;
        }
        if (kv.first == prev + 1) {
            const double d = kv.second.ms - prev_ms;
            iv.push_back(d);
            if (std::fabs(d - target) > 0.5 * period_ms) ++misses;
        }
        prev = kv.first;
        prev_ms = kv.second.ms;
    }
    if (iv.empty()) {
        std::printf("result c: no consecutive on-screen times, so no interval statistics\n");
    } else {
        double mean = 0;
        for (double d : iv) mean += d;
        mean /= iv.size();
        double var = 0;
        for (double d : iv) var += (d - mean) * (d - mean);
        std::vector<double> s = iv;
        std::sort(s.begin(), s.end());
        const double p99 = s[std::min(s.size() - 1, static_cast<size_t>(0.99 * s.size()))];
        std::printf("result c: on-screen interval over %zu consecutive pairs: target %.4f ms, mean %.4f ms, sd %.4f ms, "
                    "p50 %.4f, p99 %.4f, min %.4f, max %.4f ms; misses (|interval - target| > half a refresh) %d\n",
                    iv.size(), target, mean, std::sqrt(var / iv.size()), s[s.size() / 2], p99, s.front(), s.back(),
                    misses);
    }
    if (!o.out.empty()) {
        FILE *fo = std::fopen(o.out.c_str(), "w");
        if (fo) {
            std::fprintf(fo, "present_count,shown_ms,present_refresh\n");
            for (auto &kv : shown) std::fprintf(fo, "%u,%.4f,%u\n", kv.first, kv.second.ms, kv.second.refresh);
            std::fclose(fo);
        }
    }
    FreeLibrary(mod);
    return 0;
}
