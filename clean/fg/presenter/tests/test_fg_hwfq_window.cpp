// SPDX-License-Identifier: GPL-3.0-only
// Standalone real-window + D3D12 flip-model swap-chain repro and stress harness for FgWinCompPresenter (HWFQ)
// and its DXGI safety-net fallback. Not part of CPU ctest; run on demand through tools/gpu_slot.py.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dcomp.h>
#include <dwmapi.h>
#include <dxgi1_6.h>
#include <presentation.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "fg_dxgi_presenter.h"
#include "fg_presenter_factory.h"
#include "fg_target_scheduler.h"
#include "fg_win_common.h"
#include "fg_win_comp_presenter.h"

namespace {

using Microsoft::WRL::ComPtr;
using namespace wgfg;

constexpr FgTicks kRefresh360 = 27778;

std::atomic<const char*> g_stage{"startup"};
std::atomic<ULONGLONG> g_stage_since{0};
std::atomic<bool> g_watchdog_stop{false};
int g_checks = 0;
int g_failures = 0;

void SetStage(const char* name) {
    g_stage.store(name, std::memory_order_release);
    g_stage_since.store(GetTickCount64(), std::memory_order_release);
}

#define CHECK(cond)                                                                         \
    do {                                                                                    \
        ++g_checks;                                                                         \
        if (!(cond)) {                                                                      \
            ++g_failures;                                                                   \
            std::printf("FAIL [%s] line %d: %s\n", g_stage.load(), __LINE__, #cond);        \
            std::fflush(stdout);                                                            \
        }                                                                                   \
    } while (0)

void PumpMessages() {
    MSG msg = {};
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

void SleepPump(DWORD ms) {
    const ULONGLONG until = GetTickCount64() + ms;
    while (GetTickCount64() < until) {
        PumpMessages();
        Sleep(1);
    }
    PumpMessages();
}

LRESULT CALLBACK TestWndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_ERASEBKGND) return 1;
    return DefWindowProcW(hwnd, msg, wp, lp);
}

struct TestWindow {
    HWND hwnd = nullptr;
    HWND child = nullptr;
    UINT width = 0;
    UINT height = 0;

    bool Create(UINT w, UINT h, bool with_child = false, bool ex_topmost = true, bool no_redir = false) {
        Destroy();
        width = w;
        height = h;
        static ATOM atom = 0;
        if (!atom) {
            WNDCLASSEXW wc = {};
            wc.cbSize = sizeof(wc);
            wc.lpfnWndProc = TestWndProc;
            wc.hInstance = GetModuleHandleW(nullptr);
            wc.lpszClassName = L"WrongGpuHwfqTestWindow";
            atom = RegisterClassExW(&wc);
            if (!atom && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;
        }
        const DWORD ex = (ex_topmost ? WS_EX_TOPMOST : 0u) | (no_redir ? WS_EX_NOREDIRECTIONBITMAP : 0u);
        hwnd = CreateWindowExW(ex, L"WrongGpuHwfqTestWindow", L"WrongGPU HWFQ Test", WS_POPUP | WS_VISIBLE, 0, 0,
                               static_cast<int>(w), static_cast<int>(h), nullptr, nullptr, GetModuleHandleW(nullptr),
                               nullptr);
        if (!hwnd) return false;
        ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        if (ex_topmost) SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
        UpdateWindow(hwnd);
        if (with_child) {
            child = CreateWindowExW(WS_EX_NOREDIRECTIONBITMAP | WS_EX_TRANSPARENT, L"WrongGpuHwfqTestWindow", L"",
                                    WS_CHILD | WS_VISIBLE, 0, 0, static_cast<int>(w), static_cast<int>(h), hwnd,
                                    nullptr, GetModuleHandleW(nullptr), nullptr);
            if (!child) return false;
            ShowWindow(child, SW_SHOWNOACTIVATE);
            UpdateWindow(child);
        }
        SleepPump(30);
        return true;
    }

    void Destroy() {
        if (child) {
            DestroyWindow(child);
            child = nullptr;
        }
        if (hwnd) {
            DestroyWindow(hwnd);
            hwnd = nullptr;
        }
        PumpMessages();
    }

    ~TestWindow() { Destroy(); }
};

struct GpuHost {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> game_queue;
    ComPtr<ID3D12CommandQueue> present_queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Fence> local_fence;
    HANDLE fence_event = nullptr;
    std::uint64_t fence_value = 0;
    std::uint64_t local_fence_value = 0;
    bool tearing = false;
    LUID luid = {};

    ~GpuHost() {
        WaitIdle();
        if (fence_event) CloseHandle(fence_event);
    }

    bool Init() {
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return false;
        BOOL tear = FALSE;
        if (SUCCEEDED(factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tear, sizeof(tear))))
            tearing = tear != FALSE;

        for (UINT i = 0;; ++i) {
            ComPtr<IDXGIAdapter1> cand;
            if (factory->EnumAdapters1(i, &cand) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc = {};
            if (FAILED(cand->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            if (SUCCEEDED(D3D12CreateDevice(cand.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
                adapter = cand;
                luid = desc.AdapterLuid;
                std::printf("adapter: vendor=0x%04x device=0x%04x luid=%08lx:%08lx tearing=%d\n", desc.VendorId,
                            desc.DeviceId, static_cast<unsigned long>(luid.HighPart),
                            static_cast<unsigned long>(luid.LowPart), tearing ? 1 : 0);
                break;
            }
        }
        if (!device) return false;

        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&game_queue)))) return false;
        if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&present_queue)))) return false;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)))) return false;
        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                                             IID_PPV_ARGS(&list))) ||
            FAILED(list->Close()))
            return false;
        D3D12_DESCRIPTOR_HEAP_DESC hd = {};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        hd.NumDescriptors = 1;
        if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap)))) return false;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)))) return false;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&local_fence)))) return false;
        fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return fence_event != nullptr;
    }

    void WaitFence(std::uint64_t v) {
        if (!fence || v == 0) return;
        if (fence->GetCompletedValue() >= v) return;
        fence->SetEventOnCompletion(v, fence_event);
        WaitForSingleObject(fence_event, 2000);
    }

    void WaitLocalFence(std::uint64_t v) {
        if (!local_fence || v == 0) return;
        if (local_fence->GetCompletedValue() >= v) return;
        local_fence->SetEventOnCompletion(v, fence_event);
        WaitForSingleObject(fence_event, 2000);
    }

    void WaitIdle() {
        if (!local_fence || !game_queue || !present_queue) return;
        if (SUCCEEDED(game_queue->Signal(local_fence.Get(), ++local_fence_value))) WaitLocalFence(local_fence_value);
        if (SUCCEEDED(present_queue->Signal(local_fence.Get(), ++local_fence_value))) WaitLocalFence(local_fence_value);
    }

    std::uint64_t RenderClear(ID3D12Resource* target, float r, float g, float b,
                              D3D12_RESOURCE_STATES initial = D3D12_RESOURCE_STATE_COMMON,
                              ID3D12CommandQueue* target_q = nullptr) {
        WaitIdle();
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        D3D12_RESOURCE_BARRIER bar = {};
        bar.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        bar.Transition = {target, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, initial, D3D12_RESOURCE_STATE_RENDER_TARGET};
        list->ResourceBarrier(1, &bar);
        device->CreateRenderTargetView(target, nullptr, rtv_heap->GetCPUDescriptorHandleForHeapStart());
        const float rgba[4] = {r, g, b, 1.0f};
        list->ClearRenderTargetView(rtv_heap->GetCPUDescriptorHandleForHeapStart(), rgba, 0, nullptr);
        bar.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        bar.Transition.StateAfter = initial;
        list->ResourceBarrier(1, &bar);
        list->Close();
        ID3D12CommandList* lists[] = {list.Get()};
        ID3D12CommandQueue* q = target_q ? target_q : game_queue.Get();
        q->ExecuteCommandLists(1, lists);
        q->Signal(fence.Get(), ++fence_value);
        return fence_value;
    }

    HRESULT PresentSwapChainOnce(IDXGISwapChain4* sc, float r, float g, float b) {
        if (!sc) return E_POINTER;
        HANDLE waitable = sc->GetFrameLatencyWaitableObject();
        if (waitable) {
            WaitForSingleObject(waitable, 100);
            CloseHandle(waitable);
        }
        const UINT idx = sc->GetCurrentBackBufferIndex();
        ComPtr<ID3D12Resource> bb;
        HRESULT hr = sc->GetBuffer(idx, IID_PPV_ARGS(&bb));
        if (SUCCEEDED(hr) && bb) {
            const std::uint64_t v = RenderClear(bb.Get(), r, g, b, D3D12_RESOURCE_STATE_PRESENT, present_queue.Get());
            WaitFence(v);
        }
        const UINT pflags = tearing ? DXGI_PRESENT_ALLOW_TEARING : 0u;
        hr = sc->Present(0, pflags);
        PumpMessages();
        WaitIdle();
        return hr;
    }

    bool CreateGameSwapChain(HWND hwnd, UINT w, UINT h, DXGI_FORMAT fmt, ComPtr<IDXGISwapChain4>* out,
                             int initial_presents) {
        out->Reset();
        DXGI_SWAP_CHAIN_DESC1 sd = {};
        sd.Width = w;
        sd.Height = h;
        sd.Format = fmt;
        sd.SampleDesc.Count = 1;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.BufferCount = 3;
        sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
        sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
                   (tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0u);
        ComPtr<IDXGISwapChain1> sc1;
        HRESULT hr = factory->CreateSwapChainForHwnd(present_queue.Get(), hwnd, &sd, nullptr, nullptr, &sc1);
        if (FAILED(hr)) {
            std::printf("CreateSwapChainForHwnd failed: hr=0x%08lx\n", static_cast<unsigned long>(hr));
            return false;
        }
        factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
        if (FAILED(sc1.As(out))) return false;
        (*out)->SetMaximumFrameLatency(2);
        for (int i = 0; i < initial_presents; ++i) {
            const HRESULT phr = PresentSwapChainOnce(out->Get(), 0.1f, 0.2f, 0.4f);
            if (FAILED(phr)) {
                std::printf("initial swapchain Present failed: hr=0x%08lx\n", static_cast<unsigned long>(phr));
                return false;
            }
        }
        return true;
    }
};

struct RawCellOpts {
    const char* name = "";
    UINT width = 1920;
    UINT height = 1080;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    bool create_swapchain = false;
    int swapchain_presents = 0;
    bool use_child_hwnd = false;
    bool dcomp_with_dxgi = false;
    bool topmost = true;
    bool set_transform = true;
    bool flush11 = false;
    bool skip_ctx11_wait = false;
    bool pure_d3d11_bufs = false;
    bool force_vsync = true;
    bool target_zero = false;
    FgTicks target_offset = 80000;  // 8 ms
    bool wait_commit_init = true;
    bool detach_with_present_and_wait = false;
    bool ex_topmost = true;
    bool no_redir = false;
    bool interleave_sc = false;
};

void RunRawProbeCell(GpuHost& gpu, const RawCellOpts& opt) {
    SetStage(opt.name);
    TestWindow win;
    if (!win.Create(opt.width, opt.height, opt.use_child_hwnd, opt.ex_topmost, opt.no_redir)) {
        std::printf("cell %-48s: window create failed\n", opt.name);
        return;
    }
    ComPtr<IDXGISwapChain4> sc;
    if (opt.create_swapchain) {
        if (!gpu.CreateGameSwapChain(win.hwnd, opt.width, opt.height, opt.format, &sc, opt.swapchain_presents)) {
            std::printf("cell %-48s: swapchain create failed\n", opt.name);
            return;
        }
    }

    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    const UINT flags11 = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_PREVENT_INTERNAL_THREADING_OPTIMIZATIONS;
    ComPtr<ID3D11Device> dev11_raw;
    ComPtr<ID3D11DeviceContext> ctx11_raw;
    HRESULT hr = D3D11CreateDevice(gpu.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags11, levels, 2,
                                   D3D11_SDK_VERSION, &dev11_raw, nullptr, &ctx11_raw);
    ComPtr<ID3D11Device5> dev11;
    ComPtr<ID3D11DeviceContext4> ctx11;
    if (SUCCEEDED(hr)) hr = dev11_raw.As(&dev11);
    if (SUCCEEDED(hr)) hr = ctx11_raw.As(&ctx11);

    ComPtr<IPresentationFactory> pfac;
    ComPtr<IPresentationManager> pmgr;
    if (SUCCEEDED(hr)) hr = FgDCompCreatePresentationFactory(dev11.Get(), IID_PPV_ARGS(&pfac));
    if (SUCCEEDED(hr)) hr = pfac->CreatePresentationManager(&pmgr);

    HANDLE sh_fence = nullptr;
    ComPtr<ID3D11Fence> fence11;
    if (SUCCEEDED(hr)) hr = gpu.device->CreateSharedHandle(gpu.fence.Get(), nullptr, GENERIC_ALL, nullptr, &sh_fence);
    if (SUCCEEDED(hr)) hr = dev11->OpenSharedFence(sh_fence, IID_PPV_ARGS(&fence11));
    if (sh_fence) CloseHandle(sh_fence);

    constexpr UINT kBufs = 6;
    ComPtr<ID3D12Resource> res12[kBufs];
    ComPtr<ID3D11Texture2D> tex11[kBufs];
    ComPtr<IPresentationBuffer> pbuf[kBufs];
    HANDLE avail[kBufs] = {};
    for (UINT i = 0; i < kBufs && SUCCEEDED(hr); ++i) {
        if (opt.pure_d3d11_bufs) {
            D3D11_TEXTURE2D_DESC td = {};
            td.Width = opt.width;
            td.Height = opt.height;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = opt.format;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                           D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE;
            hr = dev11->CreateTexture2D(&td, nullptr, &tex11[i]);
        } else {
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            hp.CreationNodeMask = hp.VisibleNodeMask = 1;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = opt.width;
            rd.Height = opt.height;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.Format = opt.format;
            rd.SampleDesc.Count = 1;
            rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
            hr = gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_DISPLAY, &rd,
                                                     D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res12[i]));
            HANDLE sh = nullptr;
            if (SUCCEEDED(hr)) hr = gpu.device->CreateSharedHandle(res12[i].Get(), nullptr, GENERIC_ALL, nullptr, &sh);
            if (SUCCEEDED(hr)) hr = dev11->OpenSharedResource1(sh, IID_PPV_ARGS(&tex11[i]));
            if (sh) CloseHandle(sh);
        }
        if (SUCCEEDED(hr)) hr = pmgr->AddBufferFromResource(tex11[i].Get(), &pbuf[i]);
        if (SUCCEEDED(hr)) hr = pbuf[i]->GetAvailableEvent(&avail[i]);
    }

    HANDLE surf_handle = nullptr;
    ComPtr<IPresentationSurface> psurf;
    ComPtr<IDCompositionDesktopDevice> dcomp;
    ComPtr<IDCompositionTarget> dtarget;
    ComPtr<IDCompositionVisual2> dvis;
    ComPtr<IUnknown> dsurf;
    if (SUCCEEDED(hr)) hr = FgDCompCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, nullptr, &surf_handle);
    if (SUCCEEDED(hr)) hr = pmgr->CreatePresentationSurface(surf_handle, &psurf);
    if (SUCCEEDED(hr)) {
        psurf->SetTag(1);
        hr = psurf->SetAlphaMode(DXGI_ALPHA_MODE_IGNORE);
    }
    if (SUCCEEDED(hr)) hr = psurf->SetColorSpace(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
    const RECT src = {0, 0, static_cast<LONG>(opt.width), static_cast<LONG>(opt.height)};
    if (SUCCEEDED(hr)) hr = psurf->SetSourceRect(&src);
    if (SUCCEEDED(hr) && opt.set_transform) {
        PresentationTransform tx = {};
        tx.M11 = 1.0f;
        tx.M22 = 1.0f;
        hr = psurf->SetTransform(&tx);
    }
    ComPtr<IDXGIDevice> dxgi_dev;
    if (opt.dcomp_with_dxgi && dev11) dev11.As(&dxgi_dev);
    const HWND target_hwnd = opt.use_child_hwnd ? win.child : win.hwnd;
    if (SUCCEEDED(hr)) hr = FgDCompCreateDevice3(opt.dcomp_with_dxgi ? dxgi_dev.Get() : nullptr, IID_PPV_ARGS(&dcomp));
    if (SUCCEEDED(hr)) hr = dcomp->CreateTargetForHwnd(target_hwnd, opt.topmost ? TRUE : FALSE, &dtarget);
    if (SUCCEEDED(hr)) hr = dcomp->CreateVisual(&dvis);
    if (SUCCEEDED(hr)) hr = dcomp->CreateSurfaceFromHandle(surf_handle, &dsurf);
    if (SUCCEEDED(hr)) hr = dvis->SetContent(dsurf.Get());
    if (SUCCEEDED(hr)) hr = dtarget->SetRoot(dvis.Get());
    if (SUCCEEDED(hr)) hr = dcomp->Commit();
    if (SUCCEEDED(hr) && opt.wait_commit_init) dcomp->WaitForCommitCompletion();

    if (SUCCEEDED(hr)) hr = pmgr->ForceVSyncInterrupt(opt.force_vsync ? 1 : 0);
    if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_PresentStatus, 1);
    if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_CompositionFrame, 1);
    if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_IndependentFlipFrame, 1);

    int queued = 0, skipped = 0, canceled = 0, indep = 0, comp = 0, other_kind = 0, presented = 0;
    UINT64 last_cfid = 0;
    HRESULT last_stat_hr = S_OK;
    auto drain = [&]() {
        for (int g = 0; g < 256; ++g) {
            ComPtr<IPresentStatistics> st;
            if (!pmgr) break;
            last_stat_hr = pmgr->GetNextPresentStatistics(&st);
            if (last_stat_hr != S_OK || !st) break;
            switch (st->GetKind()) {
                case PresentStatisticsKind_PresentStatus: {
                    ComPtr<IPresentStatusPresentStatistics> ps;
                    if (SUCCEEDED(st.As(&ps))) {
                        last_cfid = ps->GetCompositionFrameId();
                        const PresentStatus v = ps->GetPresentStatus();
                        if (v == PresentStatus_Queued) ++queued;
                        else if (v == PresentStatus_Skipped) ++skipped;
                        else if (v == PresentStatus_Canceled) ++canceled;
                    }
                    break;
                }
                case PresentStatisticsKind_IndependentFlipFrame: ++indep; break;
                case PresentStatisticsKind_CompositionFrame: ++comp; break;
                default: ++other_kind; break;
            }
        }
    };

    HRESULT present_hr = S_OK;
    if (SUCCEEDED(hr)) {
        for (int f = 0; f < 16; ++f) {
            if (opt.interleave_sc && sc && (f % 4 == 0)) {
                (void)gpu.PresentSwapChainOnce(sc.Get(), 0.8f, 0.1f, 0.1f);
            }
            const UINT slot = static_cast<UINT>(f) % kBufs;
            boolean is_avail = 0;
            if (f >= static_cast<int>(kBufs)) {
                const ULONGLONG w_until = GetTickCount64() + 50;
                while (GetTickCount64() < w_until) {
                    drain();
                    PumpMessages();
                    if (SUCCEEDED(pbuf[slot]->IsAvailable(&is_avail)) && is_avail) break;
                    Sleep(1);
                }
                if (!is_avail) break;
            }
            const float shade = 0.2f + 0.04f * static_cast<float>(f % 10);
            if (opt.pure_d3d11_bufs) {
                ComPtr<ID3D11RenderTargetView> rtv;
                if (SUCCEEDED(dev11->CreateRenderTargetView(tex11[slot].Get(), nullptr, &rtv))) {
                    const float rgba[4] = {shade, 0.3f, 0.6f, 1.0f};
                    ctx11->ClearRenderTargetView(rtv.Get(), rgba);
                }
                if (opt.flush11) ctx11->Flush();
            } else {
                const std::uint64_t fv = gpu.RenderClear(res12[slot].Get(), shade, 0.3f, 0.6f);
                gpu.WaitFence(fv);
                if (!opt.skip_ctx11_wait) ctx11->Wait(fence11.Get(), fv);
                if (opt.flush11) ctx11->Flush();
            }
            present_hr = psurf->SetBuffer(pbuf[slot].Get());
            if (SUCCEEDED(present_hr)) {
                const UINT64 tt = opt.target_zero ? 0ull : static_cast<UINT64>(FgInterruptNow() + opt.target_offset);
                present_hr = pmgr->SetTargetTime(SystemInterruptTime{tt});
            }
            if (SUCCEEDED(present_hr)) present_hr = pmgr->Present();
            if (FAILED(present_hr)) break;
            ++presented;
            SleepPump(8);
            drain();
        }
        SleepPump(100);
        drain();
    }

    int avail_end = 0;
    for (UINT i = 0; i < kBufs; ++i) {
        boolean a = 0;
        if (pbuf[i] && SUCCEEDED(pbuf[i]->IsAvailable(&a)) && a) ++avail_end;
    }

    const ULONGLONG td0 = GetTickCount64();
    SetStage("raw-teardown");
    if (opt.detach_with_present_and_wait) {
        // Reproduce d6a441c order: Detach() released buffers first, then ~FgWinCompPresenter() called DetachVisual().
        for (UINT i = 0; i < kBufs; ++i) {
            pbuf[i].Reset();
            tex11[i].Reset();
            res12[i].Reset();
            if (avail[i]) {
                CloseHandle(avail[i]);
                avail[i] = nullptr;
            }
        }
        if (psurf) {
            SetStage("raw-teardown:SetBuffer(null)");
            (void)psurf->SetBuffer(nullptr);
            if (pmgr) {
                SetStage("raw-teardown:Present");
                (void)pmgr->Present();
            }
        }
        if (dvis) (void)dvis->SetContent(nullptr);
        if (dtarget) (void)dtarget->SetRoot(nullptr);
        if (dcomp) {
            SetStage("raw-teardown:Commit+Wait");
            if (SUCCEEDED(dcomp->Commit())) dcomp->WaitForCommitCompletion();
        }
    } else {
        if (dvis) (void)dvis->SetContent(nullptr);
        if (dtarget) (void)dtarget->SetRoot(nullptr);
        if (dcomp) (void)dcomp->Commit();
    }
    SetStage("raw-teardown:ReleaseCom");
    dsurf.Reset();
    dvis.Reset();
    dtarget.Reset();
    dcomp.Reset();
    psurf.Reset();
    for (UINT i = 0; i < kBufs; ++i) {
        pbuf[i].Reset();
        tex11[i].Reset();
        res12[i].Reset();
        if (avail[i]) CloseHandle(avail[i]);
    }
    pmgr.Reset();
    pfac.Reset();
    fence11.Reset();
    ctx11.Reset();
    dev11.Reset();
    ctx11_raw.Reset();
    dev11_raw.Reset();
    if (surf_handle) CloseHandle(surf_handle);
    sc.Reset();
    gpu.WaitIdle();
    const ULONGLONG td_ms = GetTickCount64() - td0;

    std::printf("cell %-48s: hr=0x%08lx phr=0x%08lx sub=%2d q=%2d skip=%2d canc=%2d indep=%2d comp=%2d oth=%d cfid=%llu avail=%d/%u td=%llums\n",
                opt.name, static_cast<unsigned long>(hr), static_cast<unsigned long>(present_hr), presented, queued,
                skipped, canceled, indep, comp, other_kind, static_cast<unsigned long long>(last_cfid), avail_end, kBufs,
                static_cast<unsigned long long>(td_ms));
    std::fflush(stdout);
}

void RunDiag(GpuHost& gpu) {
    SetStage("diag");
    DWORD sess = 999;
    ProcessIdToSessionId(GetCurrentProcessId(), &sess);
    COMPOSITION_FRAME_ID cf0 = 0;
    const HRESULT cfhr0 = FgDCompGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &cf0);
    HWND fg = GetForegroundWindow();
    wchar_t fg_title[128] = {};
    if (fg) GetWindowTextW(fg, fg_title, 127);
    std::printf("diag env: session=%lu remote=%d cfid0=%llu(0x%08lx) fg=%p '%ls'\n",
                static_cast<unsigned long>(sess), GetSystemMetrics(SM_REMOTESESSION),
                static_cast<unsigned long long>(cf0), static_cast<unsigned long>(cfhr0),
                static_cast<void*>(fg), fg_title);
    {
        COMPOSITION_FRAME_STATS fst0 = {};
        COMPOSITION_TARGET_ID tgts0[4] = {};
        UINT cnt0 = 0;
        const HRESULT shr0 = FgDCompGetStatistics(cf0, &fst0, 4, tgts0, &cnt0);
        std::printf("diag cf0 stats: hr=0x%08lx targets=%u period=%llu targetTime=%llu\n",
                    static_cast<unsigned long>(shr0), cnt0,
                    static_cast<unsigned long long>(fst0.framePeriod),
                    static_cast<unsigned long long>(fst0.targetTime));
    }

    struct Variant {
        const char* name;
        bool prime_before_commit;
        bool sc_kick_after_prime;
        bool with_sc;
        int sc_presents;
        bool fullscreen;
        bool overlapped_win;
        bool use_child;
    };
    const UINT scr_w = static_cast<UINT>(GetSystemMetrics(SM_CXSCREEN));
    const UINT scr_h = static_cast<UINT>(GetSystemMetrics(SM_CYSCREEN));
    const Variant vars[] = {
        {"A_no_sc_win", false, false, false, 0, false, true, false},
        {"A_no_sc_popup", false, false, false, 0, false, false, false},
        {"C_sc_10p_win", false, false, true, 10, false, true, false},
        {"D_sc_10p_fix", true, true, true, 10, false, true, false},
    };
    for (const Variant& v : vars) {
        SetStage(v.name);
        const ULONGLONG t0 = GetTickCount64();
        const UINT vw = v.fullscreen ? scr_w : 1280u;
        const UINT vh = v.fullscreen ? scr_h : 720u;
        TestWindow win;
        if (v.overlapped_win) {
            win.Destroy();
            static ATOM ov_atom = 0;
            if (!ov_atom) {
                WNDCLASSEXW wc = {};
                wc.cbSize = sizeof(wc);
                wc.lpfnWndProc = TestWndProc;
                wc.hInstance = GetModuleHandleW(nullptr);
                wc.lpszClassName = L"WrongGpuHwfqOvWindow";
                ov_atom = RegisterClassExW(&wc);
            }
            win.hwnd = CreateWindowExW(WS_EX_TOPMOST, L"WrongGpuHwfqOvWindow", L"WrongGPU HWFQ Ov",
                                       WS_OVERLAPPEDWINDOW | WS_VISIBLE, 100, 100, static_cast<int>(vw),
                                       static_cast<int>(vh), nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
            ShowWindow(win.hwnd, SW_SHOW);
            SetForegroundWindow(win.hwnd);
            UpdateWindow(win.hwnd);
            SleepPump(30);
        } else {
            win.Create(vw, vh, v.use_child, true, false);
        }
        DWORD cloaked = 0;
        (void)DwmGetWindowAttribute(win.hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
        (void)cloaked;
        ComPtr<IDXGISwapChain4> sc;
        if (v.with_sc) gpu.CreateGameSwapChain(win.hwnd, vw, vh, DXGI_FORMAT_R8G8B8A8_UNORM, &sc, v.sc_presents);

        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        const UINT flags11 = D3D11_CREATE_DEVICE_BGRA_SUPPORT |
                             D3D11_CREATE_DEVICE_PREVENT_INTERNAL_THREADING_OPTIMIZATIONS;
        ComPtr<ID3D11Device> d11r;
        ComPtr<ID3D11DeviceContext> c11r;
        HRESULT hr = D3D11CreateDevice(gpu.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags11, levels, 2,
                                       D3D11_SDK_VERSION, &d11r, nullptr, &c11r);
        ComPtr<ID3D11Device5> d11;
        ComPtr<ID3D11DeviceContext4> c11;
        if (SUCCEEDED(hr)) hr = d11r.As(&d11);
        if (SUCCEEDED(hr)) hr = c11r.As(&c11);
        ComPtr<IPresentationFactory> pfac;
        ComPtr<IPresentationManager> pmgr;
        if (SUCCEEDED(hr)) hr = FgDCompCreatePresentationFactory(d11.Get(), IID_PPV_ARGS(&pfac));
        if (SUCCEEDED(hr)) hr = pfac->CreatePresentationManager(&pmgr);

        HANDLE stats_ev = nullptr;
        if (SUCCEEDED(hr)) {
            if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_PresentStatus, 1);
            if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_CompositionFrame, 1);
            if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_IndependentFlipFrame, 1);
            pmgr->GetPresentStatisticsAvailableEvent(&stats_ev);
        }

        ComPtr<ID3D11Texture2D> tex[4];
        ComPtr<IPresentationBuffer> pbuf[4];
        for (int i = 0; i < 4 && SUCCEEDED(hr); ++i) {
            D3D11_TEXTURE2D_DESC td = {};
            td.Width = vw;
            td.Height = vh;
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                           D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE;
            hr = d11->CreateTexture2D(&td, nullptr, &tex[i]);
            if (SUCCEEDED(hr)) hr = pmgr->AddBufferFromResource(tex[i].Get(), &pbuf[i]);
        }

        HANDLE sh = nullptr;
        ComPtr<IPresentationSurface> psurf;
        ComPtr<IDCompositionDesktopDevice> dcomp;
        ComPtr<IDCompositionTarget> dtarget;
        ComPtr<IDCompositionVisual2> dvis;
        ComPtr<IUnknown> dsurf;
        if (SUCCEEDED(hr)) hr = FgDCompCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, nullptr, &sh);
        if (SUCCEEDED(hr)) hr = pmgr->CreatePresentationSurface(sh, &psurf);
        if (SUCCEEDED(hr)) psurf->SetTag(42);
        if (SUCCEEDED(hr)) hr = psurf->SetAlphaMode(DXGI_ALPHA_MODE_IGNORE);
        if (SUCCEEDED(hr)) hr = psurf->SetColorSpace(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
        const RECT src = {0, 0, static_cast<LONG>(vw), static_cast<LONG>(vh)};
        if (SUCCEEDED(hr)) hr = psurf->SetSourceRect(&src);
        if (SUCCEEDED(hr)) {
            PresentationTransform tx = {};
            tx.M11 = 1.0f;
            tx.M22 = 1.0f;
            hr = psurf->SetTransform(&tx);
        }
        if (SUCCEEDED(hr) && v.prime_before_commit) {
            ComPtr<ID3D11RenderTargetView> rtv;
            if (SUCCEEDED(d11->CreateRenderTargetView(tex[0].Get(), nullptr, &rtv))) {
                const float c[4] = {0.1f, 0.7f, 0.2f, 1.0f};
                c11->ClearRenderTargetView(rtv.Get(), c);
                c11->Flush();
            }
            hr = psurf->SetBuffer(pbuf[0].Get());
            if (SUCCEEDED(hr)) hr = pmgr->SetTargetTime(SystemInterruptTime{0});
            if (SUCCEEDED(hr)) hr = pmgr->Present();
        }
        ComPtr<IDXGIDevice> dxgi_dev;
        if (d11) d11.As(&dxgi_dev);
        if (SUCCEEDED(hr)) hr = FgDCompCreateDevice3(dxgi_dev.Get(), IID_PPV_ARGS(&dcomp));
        if (SUCCEEDED(hr)) hr = dcomp->CreateTargetForHwnd(v.use_child ? win.child : win.hwnd, TRUE, &dtarget);
        if (SUCCEEDED(hr)) hr = dcomp->CreateVisual(&dvis);
        if (SUCCEEDED(hr)) hr = dcomp->CreateSurfaceFromHandle(sh, &dsurf);
        if (SUCCEEDED(hr)) hr = dvis->SetContent(dsurf.Get());
        if (SUCCEEDED(hr)) hr = dtarget->SetRoot(dvis.Get());
        if (SUCCEEDED(hr)) hr = dcomp->Commit();
        if (SUCCEEDED(hr)) dcomp->WaitForCommitCompletion();
        if (SUCCEEDED(hr) && v.sc_kick_after_prime && sc) {
            (void)sc->Present(0, 0);
            PumpMessages();
        }
        if (SUCCEEDED(hr)) hr = pmgr->ForceVSyncInterrupt(1);

        const ULONGLONG t_setup = GetTickCount64() - t0;
        int q = 0, sk = 0, ca = 0, ind = 0, cmp = 0, oth = 0, ev_sig = 0;
        UINT64 cf_first = 0, cf_max = 0, inst_max = 0;
        UINT tgt_cnt_max = 0;
        UINT64 tgt_comp_time = 0;
        ComPtr<ID3D11Fence> ret_fence;
        if (SUCCEEDED(hr)) pmgr->GetPresentRetiringFence(IID_PPV_ARGS(&ret_fence));
        const UINT64 rf0 = ret_fence ? ret_fence->GetCompletedValue() : 0;

        auto inspect_cf = [&](UINT64 cf) {
            if (!cf) return;
            if (!cf_first) cf_first = cf;
            if (cf > cf_max) cf_max = cf;
            COMPOSITION_FRAME_STATS fst = {};
            COMPOSITION_TARGET_ID tgts[4] = {};
            UINT cnt = 0;
            if (SUCCEEDED(FgDCompGetStatistics(cf, &fst, 4, tgts, &cnt))) {
                if (cnt > tgt_cnt_max) tgt_cnt_max = cnt;
                if (cnt > 0) {
                    COMPOSITION_TARGET_STATS ts = {};
                    if (SUCCEEDED(FgDCompGetTargetStatistics(cf, &tgts[0], &ts)) && ts.completedStats.time)
                        tgt_comp_time = ts.completedStats.time;
                }
            }
        };

        for (int f = 0; f < 10 && SUCCEEDED(hr); ++f) {
            const int s = (f + (v.prime_before_commit ? 1 : 0)) % 4;
            ComPtr<ID3D11RenderTargetView> rtv;
            if (SUCCEEDED(d11->CreateRenderTargetView(tex[s].Get(), nullptr, &rtv))) {
                const float c[4] = {0.0f, 1.0f, 0.0f, 1.0f};
                c11->ClearRenderTargetView(rtv.Get(), c);
                c11->Flush();
            }
            hr = psurf->SetBuffer(pbuf[s].Get());
            if (SUCCEEDED(hr)) hr = pmgr->SetTargetTime(SystemInterruptTime{0});
            if (SUCCEEDED(hr)) hr = pmgr->Present();
            SleepPump(16);
            if (stats_ev && WaitForSingleObject(stats_ev, 0) == WAIT_OBJECT_0) ++ev_sig;
            for (int g = 0; g < 64; ++g) {
                ComPtr<IPresentStatistics> st;
                if (pmgr->GetNextPresentStatistics(&st) != S_OK || !st) break;
                const PresentStatisticsKind k = st->GetKind();
                if (k == PresentStatisticsKind_PresentStatus) {
                    ComPtr<IPresentStatusPresentStatistics> ps;
                    if (SUCCEEDED(st.As(&ps))) {
                        inspect_cf(ps->GetCompositionFrameId());
                        const PresentStatus pv = ps->GetPresentStatus();
                        if (pv == PresentStatus_Queued) ++q;
                        else if (pv == PresentStatus_Skipped) ++sk;
                        else if (pv == PresentStatus_Canceled) ++ca;
                    }
                } else if (k == PresentStatisticsKind_IndependentFlipFrame) {
                    ++ind;
                } else if (k == PresentStatisticsKind_CompositionFrame) {
                    ++cmp;
                    ComPtr<ICompositionFramePresentStatistics> cs;
                    if (SUCCEEDED(st.As(&cs))) {
                        UINT cnt = 0;
                        const CompositionFrameDisplayInstance* arr = nullptr;
                        cs->GetDisplayInstanceArray(&cnt, &arr);
                        if (cnt > inst_max) inst_max = cnt;
                    }
                } else {
                    ++oth;
                }
            }
        }
        SleepPump(80);
        COLORREF screen_px = 0xFFFFFFFFu;
        {
            RECT wr = {};
            GetWindowRect(win.hwnd, &wr);
            HDC sdc = GetDC(nullptr);
            if (sdc) {
                screen_px = GetPixel(sdc, wr.left + 200, wr.top + 200);
                ReleaseDC(nullptr, sdc);
            }
        }
        for (int g = 0; g < 128; ++g) {
            ComPtr<IPresentStatistics> st;
            if (!pmgr || pmgr->GetNextPresentStatistics(&st) != S_OK || !st) break;
            const PresentStatisticsKind k = st->GetKind();
            if (k == PresentStatisticsKind_PresentStatus) {
                ComPtr<IPresentStatusPresentStatistics> ps;
                if (SUCCEEDED(st.As(&ps))) {
                    inspect_cf(ps->GetCompositionFrameId());
                    const PresentStatus pv = ps->GetPresentStatus();
                    if (pv == PresentStatus_Queued) ++q;
                    else if (pv == PresentStatus_Skipped) ++sk;
                    else if (pv == PresentStatus_Canceled) ++ca;
                }
            } else if (k == PresentStatisticsKind_IndependentFlipFrame) {
                ++ind;
            } else if (k == PresentStatisticsKind_CompositionFrame) {
                ++cmp;
            } else {
                ++oth;
            }
        }
        const UINT64 rf1 = ret_fence ? ret_fence->GetCompletedValue() : 0;

        if (dvis) dvis->SetContent(nullptr);
        if (dtarget) dtarget->SetRoot(nullptr);
        if (dcomp) dcomp->Commit();
        dsurf.Reset(); dvis.Reset(); dtarget.Reset(); dcomp.Reset(); psurf.Reset();
        for (int i = 0; i < 4; ++i) { pbuf[i].Reset(); tex[i].Reset(); }
        ret_fence.Reset(); pmgr.Reset(); pfac.Reset(); c11.Reset(); d11.Reset(); c11r.Reset(); d11r.Reset();
        if (stats_ev) CloseHandle(stats_ev);
        if (sh) CloseHandle(sh);
        sc.Reset();
        const ULONGLONG t_tot = GetTickCount64() - t0;
        std::printf("diag %-15s: hr=0x%08lx px=0x%06lx setup=%llums tot=%llums q=%d sk=%d ca=%d ind=%d cmp=%d ev=%d rf=%llu->%llu tgts=%u tct=%llu inst=%llu\n",
                    v.name, static_cast<unsigned long>(hr), static_cast<unsigned long>(screen_px),
                    static_cast<unsigned long long>(t_setup),
                    static_cast<unsigned long long>(t_tot), q, sk, ca, ind, cmp, ev_sig,
                    static_cast<unsigned long long>(rf0), static_cast<unsigned long long>(rf1),
                    tgt_cnt_max, static_cast<unsigned long long>(tgt_comp_time),
                    static_cast<unsigned long long>(inst_max));
    }
}

FgIntervalPlan MakePlan(std::uint64_t epoch, std::uint64_t id, std::uint64_t frame, FgTicks base, FgTicks period,
                        std::uint32_t n, FgTicks margin, bool reanchored = false) {
    FgIntervalPlan p;
    p.epoch = epoch;
    p.interval_id = id;
    p.real_frame_id = frame;
    p.requested_count = n;
    p.generated_count = n;
    p.real_only_reason = n == 0 ? FgRealOnlyReason::MenuDisabled : FgRealOnlyReason::None;
    p.reanchored = reanchored;
    p.t_base = base;
    p.period = period;
    p.margin = margin;
    p.target_count = n + 1;
    FgBuildTargets(base, period, n, margin, p.targets);
    return p;
}

void RunPresenterProbe(GpuHost& gpu) {
    SetStage("presenter-probe");
    TestWindow win;
    CHECK(win.Create(1920, 1080, false));
    ComPtr<IDXGISwapChain4> sc;
    CHECK(gpu.CreateGameSwapChain(win.hwnd, 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, &sc, 1));

    FgCompPresenterDesc cd;
    cd.device = gpu.device.Get();
    cd.hwnd = win.hwnd;
    cd.bind_window = true;
    cd.width = 1920;
    cd.height = 1080;
    cd.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    cd.variable_refresh = true;
    cd.refresh_period = kRefresh360;
    cd.min_interval = FgVrrMinInterval(kRefresh360, false);
    cd.max_generated = 5;
    cd.slot_count = 20;
    cd.max_pending_intervals = 3;

    std::unique_ptr<FgWinCompPresenter> p;
    const FgWinResult cr = FgWinCompPresenter::Create(cd, &p);
    std::printf("presenter-probe create: %s / %s hr=0x%08lx\n", FgStatusName(cr.status), FgWinRefusalName(cr.refusal),
                static_cast<unsigned long>(cr.hr));
    CHECK(cr.Ok() && p);
    if (!p) return;

    FgWinResult fw;
    const std::uint32_t fid = p->RegisterFence(gpu.fence.Get(), &fw);
    CHECK(fid >= 1 && fw.Ok());
    CHECK(p->Attach(0xA001, 1) == FgStatus::Ok);

    FgTicks base = FgInterruptNow();
    for (std::uint64_t f = 1; f <= 20; ++f) {
        std::uint32_t slot = 0;
        const FgStatus rs = p->ReserveRealSlot(&slot);
        if (rs != FgStatus::Ok) {
            std::printf("presenter-probe frame %llu ReserveRealSlot -> %s\n", static_cast<unsigned long long>(f),
                        FgStatusName(rs));
            break;
        }
        const std::uint64_t fv = gpu.RenderClear(p->SlotResource(slot), 0.1f, 0.5f, 0.2f);
        const FgTicks now = FgInterruptNow();
        const bool reanch = f > 1 && base < now;
        if (base < now) base = now;
        const FgIntervalPlan plan = MakePlan(1, f, f, base, cd.min_interval, 0, 2500, reanch);
        base = plan.targets[0].target;
        CHECK(p->PresentRealOnly(plan, FgRealImage{1, f, FgReadyFence{fid, fv}}) == FgStatus::Ok);
        SleepPump(6);
    }
    SleepPump(120);
    const FgPresentStats st = p->Stats();
    const FgCompPresenterInfo in = p->Info();
    std::printf("presenter-probe stats: sub=%llu disp=%llu+%llu skip=%llu canc=%llu indep=%llu comp=%llu q=%llu "
                "sk=%llu ca=%llu wd=%llu vis=%llu\n",
                static_cast<unsigned long long>(st.real_submitted), static_cast<unsigned long long>(st.displayed_real),
                static_cast<unsigned long long>(st.displayed_generated),
                static_cast<unsigned long long>(st.display_skipped), static_cast<unsigned long long>(st.cancelled),
                static_cast<unsigned long long>(in.indep_flip_frames), static_cast<unsigned long long>(in.comp_frames),
                static_cast<unsigned long long>(in.status_queued), static_cast<unsigned long long>(in.status_skipped),
                static_cast<unsigned long long>(in.status_canceled),
                static_cast<unsigned long long>(in.watchdog_retired),
                static_cast<unsigned long long>(in.visible_submitted));
    std::fflush(stdout);

    const ULONGLONG td0 = GetTickCount64();
    SetStage("presenter-probe:Detach");
    FgStatus ds = p->Detach();
    for (int i = 0; i < 100 && ds == FgStatus::Busy; ++i) {
        SleepPump(5);
        ds = p->Detach();
    }
    std::printf("presenter-probe Detach -> %s (%llums)\n", FgStatusName(ds),
                static_cast<unsigned long long>(GetTickCount64() - td0));
    std::fflush(stdout);
    SetStage("presenter-probe:Destroy");
    p.reset();
    std::printf("presenter-probe Destroy ok (%llums)\n", static_cast<unsigned long long>(GetTickCount64() - td0));
    std::fflush(stdout);
}

void EnablePerMonitorDpiV2() {
    using SetDpiCtxFn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
    if (const HMODULE u32 = GetModuleHandleW(L"user32.dll")) {
        if (auto fn = reinterpret_cast<SetDpiCtxFn>(reinterpret_cast<void*>(
                GetProcAddress(u32, "SetProcessDpiAwarenessContext")))) {
            fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
    }
}

bool IsGreenPixel(COLORREF px) {
    return px != CLR_INVALID && GetGValue(px) >= 0xC0 && GetRValue(px) <= 0x40 && GetBValue(px) <= 0x40;
}

long long SignedPercentileUs(std::vector<FgTicks> v, int pct) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const std::size_t idx = (v.size() - 1) * static_cast<std::size_t>(pct) / 100;
    return static_cast<long long>(v[idx] / 10);
}

long long AbsMaxUs(const std::vector<FgTicks>& v) {
    FgTicks m = 0;
    for (FgTicks x : v) {
        const FgTicks a = x < 0 ? -x : x;
        if (a > m) m = a;
    }
    return static_cast<long long>(m / 10);
}

void RunFullscreenPaced240Gate(GpuHost& gpu) {
    SetStage("fs-240hz-gate");
    EnablePerMonitorDpiV2();
    const UINT scr_w = static_cast<UINT>(GetSystemMetrics(SM_CXSCREEN));
    const UINT scr_h = static_cast<UINT>(GetSystemMetrics(SM_CYSCREEN));

    TestWindow win;
    CHECK(win.Create(scr_w, scr_h, false, false, false));
    ShowWindow(win.hwnd, SW_SHOW);
    SetForegroundWindow(win.hwnd);
    UpdateWindow(win.hwnd);
    SleepPump(40);

    FgCompPresenterDesc cd;
    cd.device = gpu.device.Get();
    cd.hwnd = win.hwnd;
    cd.bind_window = true;
    cd.width = scr_w;
    cd.height = scr_h;
    cd.format = DXGI_FORMAT_B8G8R8A8_UNORM;
    cd.variable_refresh = true;
    cd.refresh_period = kRefresh360;
    cd.min_interval = FgVrrMinInterval(kRefresh360, false);
    cd.max_generated = 5;
    cd.slot_count = 16;
    cd.max_pending_intervals = 3;

    std::unique_ptr<FgWinCompPresenter> p;
    const FgWinResult cr = FgWinCompPresenter::Create(cd, &p);
    CHECK(cr.Ok() && p);
    if (!p) return;

    FgWinResult fw;
    const std::uint32_t fid = p->RegisterFence(gpu.fence.Get(), &fw);
    CHECK(fid >= 1 && fw.Ok());
    CHECK(p->Attach(0xB100, 1) == FgStatus::Ok);

    struct ErrTrack {
        std::mutex mu;
        std::vector<FgTicks> errs;
        std::vector<FgTicks> sub_errs;
        std::vector<FgTicks> sub_gaps;
        FgTicks last_sub = 0;
    } err_track;
    p->SetTraceCallback(
        [](void* ctx, std::uint64_t, std::uint64_t, std::uint32_t, bool, std::uint8_t kind, FgTicks ts,
           FgTicks target) {
            auto* et = static_cast<ErrTrack*>(ctx);
            std::lock_guard<std::mutex> lock(et->mu);
            if (kind == 1 && ts > 0 && target > 0) {
                et->sub_errs.push_back(ts - target);
                if (et->last_sub > 0) et->sub_gaps.push_back(ts - et->last_sub);
                et->last_sub = ts;
            } else if (kind == 2 && ts > 0 && target > 0) {
                et->errs.push_back(ts - target);
            }
        },
        &err_track);

    // 2.5 seconds at 240 Hz = 600 frames on 2560x1440 opaque fullscreen borderless popup.
    constexpr int kFrames = 600;
    constexpr FgTicks kStep240 = 41667;
    FgHighResTimer pace;
    FgTicks next_target = FgInterruptNow() + 60000;
    for (int f = 1; f <= kFrames; ++f) {
        SetStage("fs-240hz-present");
        const FgTicks wake_at = next_target - 25000;
        if (FgInterruptNow() < wake_at && pace.ArmAt(wake_at)) WaitForSingleObject(pace.Handle(), 15);
        PumpMessages();
        std::uint32_t slot = 0;
        FgStatus rs = p->ReserveRealSlot(&slot);
        for (int w = 0; w < 20 && rs == FgStatus::NoSlot; ++w) {
            SleepPump(1);
            rs = p->ReserveRealSlot(&slot);
        }
        CHECK(rs == FgStatus::Ok);
        if (rs == FgStatus::Ok) {
            const std::uint64_t fv = gpu.RenderClear(p->SlotResource(slot), 0.0f, 1.0f, 0.0f);
            const FgTicks base = next_target - kStep240;
            const FgIntervalPlan plan = MakePlan(1, static_cast<std::uint64_t>(f), static_cast<std::uint64_t>(f),
                                                 base, kStep240, 0, 2500, f > 1);
            CHECK(p->PresentRealOnly(plan, FgRealImage{1, static_cast<std::uint64_t>(f), FgReadyFence{fid, fv}}) ==
                  FgStatus::Ok);
        }
        next_target += kStep240;
    }
    SleepPump(80);
    COLORREF px = 0xFFFFFFFFu;
    {
        HDC sdc = GetDC(nullptr);
        if (sdc) {
            px = GetPixel(sdc, 200, 200);
            ReleaseDC(nullptr, sdc);
        }
    }
    const FgPresentStats st = p->Stats();
    const FgCompPresenterInfo in = p->Info();
    const FgDisplaySummary disp = p->ReadDisplayCadence(false);
    std::vector<FgTicks> errs, sub_errs, sub_gaps;
    {
        std::lock_guard<std::mutex> lock(err_track.mu);
        errs = err_track.errs;
        sub_errs = err_track.sub_errs;
        sub_gaps = err_track.sub_gaps;
    }
    const long long err_p50 = SignedPercentileUs(errs, 50);
    const long long err_p95 = SignedPercentileUs(errs, 95);
    const long long err_max = AbsMaxUs(errs);
    const long long sub_p50 = SignedPercentileUs(sub_errs, 50);
    const long long sub_p95 = SignedPercentileUs(sub_errs, 95);
    const long long gap_p05 = SignedPercentileUs(sub_gaps, 5);
    const double skip_pct =
        (st.displayed_real + st.display_skipped) > 0
            ? (100.0 * static_cast<double>(st.display_skipped) /
               static_cast<double>(st.displayed_real + st.display_skipped))
            : 0.0;
    std::printf("fs-240hz-gate (%ux%u): px=0x%06lx disp=%llu skip=%llu(%.2f%%) grid=%lldus bumps=%llu "
                "sub_err(p50/p95)=%lld/%lldus gap_p05=%lldus | dt_p50=%lldus err_us(p50/p95/max)=%lld/%lld/%lld\n",
                scr_w, scr_h, static_cast<unsigned long>(px),
                static_cast<unsigned long long>(st.displayed_real),
                static_cast<unsigned long long>(st.display_skipped), skip_pct,
                static_cast<long long>(in.grid.period / 10),
                static_cast<unsigned long long>(in.grid_phase_bumps),
                sub_p50, sub_p95, gap_p05,
                static_cast<long long>(disp.interval_p50 / 10), err_p50, err_p95, err_max);
    std::fflush(stdout);

    CHECK(IsGreenPixel(px));
    CHECK(st.displayed_real >= 550);
    CHECK(st.display_skipped * 20 <= st.displayed_real);  // <= 5% skips
    CHECK(st.display_time_measured);
    CHECK(in.display_time_iflip + in.display_time_comp + in.display_time_dcomp_stats >= 550);
    CHECK(std::abs(err_p50) <= 5000);
    CHECK(std::abs(err_p95) <= 12000);
    CHECK(st.refused_calls == 0);
    CHECK(in.watchdog_retired == 0);

    SetStage("fs-240hz-gate:Detach");
    FgStatus ds = p->Detach();
    for (int i = 0; i < 100 && ds == FgStatus::Busy; ++i) {
        SleepPump(5);
        ds = p->Detach();
    }
    CHECK(ds == FgStatus::Ok);
    p.reset();
    gpu.WaitIdle();
}

void RunLongHwfqRun(GpuHost& gpu) {
    SetStage("long-hwfq-run");
    EnablePerMonitorDpiV2();
    const UINT scr_w = static_cast<UINT>(GetSystemMetrics(SM_CXSCREEN));
    const UINT scr_h = static_cast<UINT>(GetSystemMetrics(SM_CYSCREEN));
    TestWindow win;
    CHECK(win.Create(scr_w, scr_h, false, false, false));
    ShowWindow(win.hwnd, SW_SHOW);
    SetForegroundWindow(win.hwnd);
    UpdateWindow(win.hwnd);
    SleepPump(40);
    ComPtr<IDXGISwapChain4> sc;
    CHECK(gpu.CreateGameSwapChain(win.hwnd, scr_w, scr_h, DXGI_FORMAT_R8G8B8A8_UNORM, &sc, 2));

    FgCompPresenterDesc cd;
    cd.device = gpu.device.Get();
    cd.hwnd = win.hwnd;
    cd.bind_window = true;
    cd.width = scr_w;
    cd.height = scr_h;
    cd.format = DXGI_FORMAT_R8G8B8A8_UNORM;
    cd.variable_refresh = true;
    cd.refresh_period = kRefresh360;
    cd.min_interval = FgVrrMinInterval(kRefresh360, false);
    cd.max_generated = 5;
    cd.slot_count = 20;
    cd.max_pending_intervals = 3;

    std::unique_ptr<FgWinCompPresenter> p;
    const FgWinResult cr = FgWinCompPresenter::Create(cd, &p);
    CHECK(cr.Ok() && p);
    if (!p) return;

    // Startup chain kick matching TryCreateSession in fg_asi_session.cpp.
    if (sc) {
        HANDLE w = sc->GetFrameLatencyWaitableObject();
        if (w) {
            WaitForSingleObject(w, 50);
            CloseHandle(w);
        }
        (void)sc->Present(0, 0);
        PumpMessages();
    }

    FgWinResult fw;
    const std::uint32_t fid = p->RegisterFence(gpu.fence.Get(), &fw);
    CHECK(fid >= 1 && fw.Ok());
    CHECK(p->Attach(0xB001, 1) == FgStatus::Ok);

    struct ErrTrack {
        std::mutex mu;
        std::vector<FgTicks> errs;
        std::vector<FgTicks> sub_errs;
        std::vector<FgTicks> sub_gaps;
        FgTicks last_sub = 0;
    } err_track;
    p->SetTraceCallback(
        [](void* ctx, std::uint64_t, std::uint64_t, std::uint32_t, bool, std::uint8_t kind, FgTicks ts,
           FgTicks target) {
            auto* et = static_cast<ErrTrack*>(ctx);
            std::lock_guard<std::mutex> lock(et->mu);
            if (kind == 1 && ts > 0 && target > 0) {
                et->sub_errs.push_back(ts - target);
                if (et->last_sub > 0) et->sub_gaps.push_back(ts - et->last_sub);
                et->last_sub = ts;
            } else if (kind == 2 && ts > 0 && target > 0) {
                et->errs.push_back(ts - target);
            }
        },
        &err_track);

    FgSchedulerConfig scfg;
    scfg.max_generated = 5;
    scfg.adaptive_count = true;
    scfg.bridge_and_submit_margin = 2500;
    scfg.first_image_budget = 40000;
    scfg.max_lead = 500000;
    scfg.display.min_interval = cd.min_interval;
    scfg.display.max_interval = 208333;
    scfg.display.margin = 200;
    scfg.display.hysteresis = 400;
    scfg.display.exit_intervals = 2;
    scfg.display.reenter_intervals = 3;
    FgTargetScheduler sched(scfg);
    CHECK(sched.BeginEpoch(1) == FgStatus::Ok);

    // 4 phases: real-only warmup (16 frames), 2x (50 frames), 3x (50 frames), 4x (50 frames) at ~60 fps (~16.67 ms).
    struct Phase {
        const char* name;
        std::uint32_t req_n;
        bool gen_on;
        int frames;
    };
    const Phase phases[] = {
        {"warmup-real", 0, false, 16},
        {"2x", 1, true, 50},
        {"3x", 2, true, 50},
        {"4x", 3, true, 50},
    };
    std::uint64_t frame_id = 0;
    constexpr FgTicks kBasePeriod = 166667;  // 60 fps
    FgTicks next_arrival = FgInterruptNow();
    for (const Phase& ph : phases) {
        for (int i = 0; i < ph.frames; ++i) {
            SetStage(ph.name);
            while (FgInterruptNow() < next_arrival) {
                PumpMessages();
                Sleep(1);
            }
            PumpMessages();
            const FgTicks now = FgInterruptNow();
            FgGameFrame gf;
            gf.epoch = 1;
            gf.frame_id = ++frame_id;
            gf.arrival = now;
            gf.budget_origin = now;
            gf.requested_count = ph.req_n;
            gf.generation_enabled = ph.gen_on;
            FgIntervalPlan plan;
            CHECK(sched.Commit(gf, &plan) == FgStatus::Ok);
            if (plan.generated_count == 0) {
                std::uint32_t slot = 0;
                const FgStatus rs = p->ReserveRealSlot(&slot);
                CHECK(rs == FgStatus::Ok);
                if (rs == FgStatus::Ok) {
                    const std::uint64_t fv = gpu.RenderClear(p->SlotResource(slot), 0.0f, 1.0f, 0.0f);
                    CHECK(p->PresentRealOnly(plan, FgRealImage{1, gf.frame_id, FgReadyFence{fid, fv}}) == FgStatus::Ok);
                }
            } else {
                FgIntervalReservation res;
                const FgStatus bs = p->BeginInterval(plan, &res);
                CHECK(bs == FgStatus::Ok);
                if (bs == FgStatus::Ok) {
                    FgIntervalSubmission sub;
                    sub.epoch = 1;
                    sub.interval_id = plan.interval_id;
                    sub.count = res.count;
                    for (std::uint32_t k = 0; k < res.count; ++k) {
                        if (!res.slots[k].reserved) continue;
                        const float g = res.slots[k].real ? 1.0f : 0.9f;
                        const std::uint64_t fv = gpu.RenderClear(p->SlotResource(res.slots[k].slot), 0.0f, g, 0.0f);
                        sub.ready[k] = FgReadyFence{fid, fv};
                        sub.produced[k] = true;
                    }
                    CHECK(p->SubmitInterval(sub) == FgStatus::Ok);
                }
            }
            next_arrival += kBasePeriod;
        }
    }
    SleepPump(120);
    COLORREF px = 0xFFFFFFFFu;
    {
        HDC sdc = GetDC(nullptr);
        if (sdc) {
            px = GetPixel(sdc, 200, 200);
            ReleaseDC(nullptr, sdc);
        }
    }
    const FgPresentStats st = p->Stats();
    const FgCompPresenterInfo in = p->Info();
    const FgDisplaySummary disp = p->ReadDisplayCadence(false);
    std::vector<FgTicks> errs;
    {
        std::lock_guard<std::mutex> lock(err_track.mu);
        errs = err_track.errs;
    }
    const long long err_p50 = SignedPercentileUs(errs, 50);
    const long long err_p95 = SignedPercentileUs(errs, 95);
    const long long err_max = AbsMaxUs(errs);
    const std::uint64_t total_disp = st.displayed_real + st.displayed_generated;
    const double skip_pct =
        (total_disp + st.display_skipped) > 0
            ? (100.0 * static_cast<double>(st.display_skipped) /
               static_cast<double>(total_disp + st.display_skipped))
            : 0.0;
    std::printf("long-hwfq-run: px=0x%06lx real_disp=%llu gen_disp=%llu gen_sched=%llu late=%llu skip=%llu(%.2f%%) "
                "grid=%lldus bumps=%llu short=%llu | dt_p50=%lldus lat_p50=%lldus "
                "err_us(p50/p95/max)=%lld/%lld/%lld\n",
                static_cast<unsigned long>(px),
                static_cast<unsigned long long>(st.displayed_real),
                static_cast<unsigned long long>(st.displayed_generated),
                static_cast<unsigned long long>(st.generated_scheduled),
                static_cast<unsigned long long>(st.dropped_late),
                static_cast<unsigned long long>(st.display_skipped), skip_pct,
                static_cast<long long>(in.grid.period / 10),
                static_cast<unsigned long long>(in.grid_phase_bumps),
                static_cast<unsigned long long>(disp.short_intervals),
                static_cast<long long>(disp.interval_p50 / 10), static_cast<long long>(disp.latency_p50 / 10),
                err_p50, err_p95, err_max);
    std::fflush(stdout);

    CHECK(IsGreenPixel(px));
    CHECK(st.displayed_real >= 150);
    CHECK(st.displayed_generated >= 270);
    CHECK(st.display_skipped * 10 <= total_disp);  // <= 10% skip ratio
    CHECK(in.display_time_iflip + in.display_time_comp + in.display_time_dcomp_stats >= 400);
    CHECK(st.display_time_measured);
    CHECK(std::abs(err_p50) <= 5000);
    CHECK(std::abs(err_p95) <= 15000);
    CHECK(st.refused_calls == 0);
    CHECK(in.watchdog_retired == 0);

    SetStage("long-hwfq-run:Detach");
    FgStatus ds = p->Detach();
    for (int i = 0; i < 100 && ds == FgStatus::Busy; ++i) {
        SleepPump(5);
        ds = p->Detach();
    }
    CHECK(ds == FgStatus::Ok);
    p.reset();
    gpu.WaitIdle();
}

void RunJitteredBaseGate(GpuHost& gpu) {
    SetStage("jittered-base-gate");
    EnablePerMonitorDpiV2();
    const UINT scr_w = static_cast<UINT>(GetSystemMetrics(SM_CXSCREEN));
    const UINT scr_h = static_cast<UINT>(GetSystemMetrics(SM_CYSCREEN));

    TestWindow win;
    CHECK(win.Create(scr_w, scr_h, false, false, false));
    ShowWindow(win.hwnd, SW_SHOW);
    SetForegroundWindow(win.hwnd);
    UpdateWindow(win.hwnd);
    SleepPump(40);
    ComPtr<IDXGISwapChain4> sc;
    CHECK(gpu.CreateGameSwapChain(win.hwnd, scr_w, scr_h, DXGI_FORMAT_R8G8B8A8_UNORM, &sc, 2));

    FgClockMap clk;
    FgRefreshGrid live_grid;
    const bool got_grid = FgReadCompositorGrid(clk, 0, &live_grid) && live_grid.period >= 10000 && live_grid.period <= 416667;
    const FgTicks comp_period = got_grid ? live_grid.period : kRefresh360;
    const double comp_hz = 10000000.0 / static_cast<double>(comp_period);
    std::printf("jittered-base-gate: screen=%ux%u live_comp=%.2fHz (%lldus)\n",
                scr_w, scr_h, comp_hz, static_cast<long long>(comp_period / 10));

    // Dedicated shared fence signaled asynchronously to simulate realistic GPU executor readiness lag + occasional
    // late bursts (2.2..4.6 ms normal, 6.8 ms spike every 12th frame) after RenderClear finishes on GPU.
    ComPtr<ID3D12Fence> sim_fence;
    CHECK(SUCCEEDED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&sim_fence))));
    if (!sim_fence) return;

    struct DelayItem {
        std::uint64_t gpu_val = 0;
        std::uint64_t sim_val = 0;
        FgTicks signal_at = 0;
    };
    struct LagThread {
        GpuHost* gpu = nullptr;
        ID3D12Fence* sim = nullptr;
        std::mutex mu;
        std::deque<DelayItem> q;
        HANDLE wake = nullptr;
        HANDLE stop = nullptr;
        std::thread th;
        FgHighResTimer timer;

        void Start(GpuHost* g, ID3D12Fence* s) {
            gpu = g;
            sim = s;
            wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            stop = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            th = std::thread([this]() {
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
                HANDLE hs[3] = {stop, wake, timer.Handle()};
                for (;;) {
                    DelayItem item;
                    bool has_item = false;
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        if (!q.empty()) {
                            item = q.front();
                            has_item = true;
                        }
                    }
                    if (!has_item) {
                        const DWORD w = WaitForMultipleObjects(2, hs, FALSE, 50);
                        if (w == WAIT_OBJECT_0) break;
                        continue;
                    }
                    const FgTicks now = FgInterruptNow();
                    if (now < item.signal_at) {
                        if (timer.Valid() && timer.ArmAt(item.signal_at)) {
                            const DWORD w = WaitForMultipleObjects(3, hs, FALSE, 20);
                            if (w == WAIT_OBJECT_0) break;
                        } else {
                            Sleep(1);
                        }
                        continue;
                    }
                    while (gpu->fence->GetCompletedValue() < item.gpu_val) {
                        if (WaitForSingleObject(stop, 0) == WAIT_OBJECT_0) return;
                        Sleep(0);
                    }
                    sim->Signal(item.sim_val);
                    {
                        std::lock_guard<std::mutex> lock(mu);
                        if (!q.empty() && q.front().sim_val == item.sim_val) q.pop_front();
                    }
                }
            });
        }
        void Push(std::uint64_t gpu_val, std::uint64_t sim_val, FgTicks signal_at) {
            {
                std::lock_guard<std::mutex> lock(mu);
                q.push_back(DelayItem{gpu_val, sim_val, signal_at});
            }
            if (wake) SetEvent(wake);
        }
        void Stop() {
            if (stop) SetEvent(stop);
            if (th.joinable()) th.join();
            if (wake) CloseHandle(wake);
            if (stop) CloseHandle(stop);
            wake = stop = nullptr;
        }
    } lag_thread;
    lag_thread.Start(&gpu, sim_fence.Get());
    std::uint64_t sim_val = 0;

    // Deterministic 90-130 fps jittered frame periods (centered ~108 fps = 9.26 ms, +/- 19% realistic swing).
    const FgTicks jitter_periods[16] = {
        92600, 81500, 106500, 78500, 99500, 86500, 110500, 80500,
        95500, 84500, 104500, 90500, 77500, 108500, 88500, 94500,
    };
    const FgTicks lag_table[12] = {
        24000, 31000, 27000, 38000, 29000, 35000, 42000, 26000, 33000, 45000, 30000, 68000,
    };

    for (std::uint32_t req_n = 1; req_n <= 5; ++req_n) {
        char stg[48];
        std::snprintf(stg, sizeof stg, "jittered-x%u", req_n + 1);
        SetStage(stg);

        FgCompPresenterDesc cd;
        cd.device = gpu.device.Get();
        cd.hwnd = win.hwnd;
        cd.bind_window = true;
        cd.width = scr_w;
        cd.height = scr_h;
        cd.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        cd.variable_refresh = true;
        cd.refresh_period = comp_period;
        cd.min_interval = FgVrrMinInterval(comp_period, true);
        cd.max_generated = 5;
        cd.slot_count = 20;
        cd.max_pending_intervals = 3;

        std::unique_ptr<FgWinCompPresenter> p;
        const FgWinResult cr = FgWinCompPresenter::Create(cd, &p);
        CHECK(cr.Ok() && p);
        if (!p) break;

        if (sc) {
            HANDLE w = sc->GetFrameLatencyWaitableObject();
            if (w) {
                WaitForSingleObject(w, 50);
                CloseHandle(w);
            }
            (void)sc->Present(0, 0);
            PumpMessages();
        }

        FgWinResult fw;
        const std::uint32_t fid = p->RegisterFence(sim_fence.Get(), &fw);
        CHECK(fid >= 1 && fw.Ok());
        CHECK(p->Attach(0xB400 + req_n, 1) == FgStatus::Ok);

        struct ErrTrack {
            std::mutex mu;
            std::vector<FgTicks> errs;
        } err_track;
        p->SetTraceCallback(
            [](void* ctx, std::uint64_t, std::uint64_t, std::uint32_t, bool, std::uint8_t kind, FgTicks ts,
               FgTicks target) {
                if (kind == 2 && ts > 0 && target > 0) {
                    auto* et = static_cast<ErrTrack*>(ctx);
                    std::lock_guard<std::mutex> lock(et->mu);
                    et->errs.push_back(ts - target);
                }
            },
            &err_track);

        FgSchedulerConfig scfg;
        scfg.max_generated = 5;
        scfg.adaptive_count = true;
        scfg.bridge_and_submit_margin = 2500;
        scfg.first_image_budget = 60000;  // 6.0 ms (matches Cyberpunk C preset settled budget)
        scfg.max_lead = 500000;
        scfg.display.min_interval = cd.min_interval;
        scfg.display.max_interval = 208333;
        scfg.display.margin = 200;
        scfg.display.hysteresis = 400;
        scfg.display.exit_intervals = 2;
        scfg.display.reenter_intervals = 3;
        scfg.guard_overflow_delivery = true;
        FgTargetScheduler sched(scfg);
        CHECK(sched.BeginEpoch(1) == FgStatus::Ok);

        // 16 warmup frames + 150 measured jittered frames (~1.5 s per multiplier).
        constexpr int kWarmup = 16;
        constexpr int kMeasured = 150;
        FgHighResTimer pace;
        FgTicks next_arrival = FgInterruptNow() + 10000;
        FgTicks t_meas_start = 0;
        FgTicks t_meas_end = 0;
        FgPresentStats base_st{};
        FgCompPresenterInfo base_in{};
        std::uint64_t fid_seq = 0;

        for (int f = 0; f < kWarmup + kMeasured; ++f) {
            SetStage(stg);
            if (FgInterruptNow() < next_arrival && pace.ArmAt(next_arrival))
                WaitForSingleObject(pace.Handle(), 20);
            PumpMessages();
            const FgTicks now = FgInterruptNow();
            if (f == kWarmup) {
                t_meas_start = now;
                base_st = p->Stats();
                base_in = p->Info();
                std::lock_guard<std::mutex> lock(err_track.mu);
                err_track.errs.clear();
            }

            FgGameFrame gf;
            gf.epoch = 1;
            gf.frame_id = ++fid_seq;
            gf.arrival = now;
            gf.budget_origin = now;
            gf.requested_count = f < kWarmup ? 0u : req_n;
            gf.generation_enabled = f >= kWarmup;
            FgIntervalPlan plan;
            CHECK(sched.Commit(gf, &plan) == FgStatus::Ok);

            const FgTicks lag = f < kWarmup ? 15000 : lag_table[f % 12];
            if (plan.generated_count == 0) {
                std::uint32_t slot = 0;
                FgStatus rs = p->ReserveRealSlot(&slot);
                for (int w = 0; w < 20 && rs == FgStatus::NoSlot; ++w) {
                    SleepPump(1);
                    rs = p->ReserveRealSlot(&slot);
                }
                CHECK(rs == FgStatus::Ok);
                if (rs == FgStatus::Ok) {
                    const std::uint64_t gfv = gpu.RenderClear(p->SlotResource(slot), 0.0f, 1.0f, 0.0f);
                    const std::uint64_t sv = ++sim_val;
                    lag_thread.Push(gfv, sv, now + lag);
                    CHECK(p->PresentRealOnly(plan, FgRealImage{1, gf.frame_id, FgReadyFence{fid, sv}}) == FgStatus::Ok);
                }
            } else {
                FgIntervalReservation res;
                FgStatus bs = p->BeginInterval(plan, &res);
                for (int w = 0; w < 20 && bs == FgStatus::NoSlot; ++w) {
                    SleepPump(1);
                    bs = p->BeginInterval(plan, &res);
                }
                CHECK(bs == FgStatus::Ok);
                if (bs == FgStatus::Ok) {
                    FgIntervalSubmission sub;
                    sub.epoch = 1;
                    sub.interval_id = plan.interval_id;
                    sub.count = res.count;
                    for (std::uint32_t k = 0; k < res.count; ++k) {
                        if (!res.slots[k].reserved) continue;
                        const float g = res.slots[k].real ? 1.0f : 0.9f;
                        const std::uint64_t gfv = gpu.RenderClear(p->SlotResource(res.slots[k].slot), 0.0f, g, 0.0f);
                        const std::uint64_t sv = ++sim_val;
                        // All images of the interval share realistic GPU readiness lag (staggered slightly per index).
                        const FgTicks ready_t = now + lag + static_cast<FgTicks>(k) * 3000;
                        lag_thread.Push(gfv, sv, ready_t);
                        sub.ready[k] = FgReadyFence{fid, sv};
                        sub.produced[k] = true;
                    }
                    CHECK(p->SubmitInterval(sub) == FgStatus::Ok);
                }
            }
            next_arrival += jitter_periods[f % 16];
        }
        t_meas_end = FgInterruptNow();
        SleepPump(100);

        COLORREF px = 0xFFFFFFFFu;
        {
            HDC sdc = GetDC(nullptr);
            if (sdc) {
                px = GetPixel(sdc, 200, 200);
                ReleaseDC(nullptr, sdc);
            }
        }
        const FgPresentStats st = p->Stats();
        const FgCompPresenterInfo in = p->Info();
        const FgDisplaySummary disp = p->ReadDisplayCadence(false);
        std::vector<FgTicks> errs;
        {
            std::lock_guard<std::mutex> lock(err_track.mu);
            errs = err_track.errs;
        }
        const std::uint64_t d_real = st.displayed_real - base_st.displayed_real;
        const std::uint64_t d_gen = st.displayed_generated - base_st.displayed_generated;
        const std::uint64_t d_skip = st.display_skipped - base_st.display_skipped;
        const std::uint64_t d_late = st.dropped_late - base_st.dropped_late;
        const std::uint64_t d_press = st.dropped_pressure - base_st.dropped_pressure;
        const std::uint64_t d_comp_l = in.composed_drop_late - base_in.composed_drop_late;
        const std::uint64_t d_comp_c = in.composed_drop_collision - base_in.composed_drop_collision;
        const std::uint64_t d_bumps = in.grid_phase_bumps - base_in.grid_phase_bumps;
        const std::uint64_t d_tot = d_real + d_gen;
        const double skip_pct =
            (d_tot + d_skip) > 0 ? (100.0 * static_cast<double>(d_skip) / static_cast<double>(d_tot + d_skip)) : 0.0;
        const double secs = static_cast<double>(t_meas_end - t_meas_start) / 10000000.0;
        const double base_fps = secs > 0.0 ? static_cast<double>(kMeasured) / secs : 0.0;
        const double disp_fps = secs > 0.0 ? static_cast<double>(d_tot) / secs : 0.0;
        const double mult = d_real > 0 ? static_cast<double>(d_tot) / static_cast<double>(d_real) : 0.0;
        const long long err_p50 = SignedPercentileUs(errs, 50);
        const long long err_p95 = SignedPercentileUs(errs, 95);
        const long long err_max = AbsMaxUs(errs);

        std::printf("jittered x%u: px=0x%06lx base=%.1ffps disp=%.1ffps(%.2fx) real=%llu gen=%llu "
                    "drop(l=%llu/p=%llu,comp_l=%llu/c=%llu) skip=%llu(%.2f%%) short=%llu bumps=%llu | dt_p50=%lldus "
                    "err_us(p50/p95/max)=%lld/%lld/%lld meas=%d wd=%llu ref=%llu\n",
                    req_n + 1, static_cast<unsigned long>(px), base_fps, disp_fps, mult,
                    static_cast<unsigned long long>(d_real), static_cast<unsigned long long>(d_gen),
                    static_cast<unsigned long long>(d_late), static_cast<unsigned long long>(d_press),
                    static_cast<unsigned long long>(d_comp_l), static_cast<unsigned long long>(d_comp_c),
                    static_cast<unsigned long long>(d_skip), skip_pct,
                    static_cast<unsigned long long>(disp.short_intervals),
                    static_cast<unsigned long long>(d_bumps),
                    static_cast<long long>(disp.interval_p50 / 10), err_p50, err_p95, err_max,
                    st.display_time_measured ? 1 : 0, static_cast<unsigned long long>(in.watchdog_retired),
                    static_cast<unsigned long long>(st.refused_calls));
        std::fflush(stdout);

        CHECK(IsGreenPixel(px));
        CHECK(skip_pct < 2.0);
        CHECK(std::abs(err_p95) <= static_cast<long long>(comp_period / 10) + 1200);
        CHECK(st.display_time_measured);
        CHECK(st.refused_calls == 0);
        CHECK(in.watchdog_retired == 0);
        CHECK(d_real >= 135);
        CHECK(disp_fps >= (req_n == 1 ? 185.0 : 270.0));

        SetStage("jittered:Detach");
        FgStatus ds = p->Detach();
        for (int i = 0; i < 100 && ds == FgStatus::Busy; ++i) {
            SleepPump(5);
            ds = p->Detach();
        }
        CHECK(ds == FgStatus::Ok);
        p.reset();
        gpu.WaitIdle();
    }

    lag_thread.Stop();
    gpu.WaitIdle();
}

void RunCyclesStress(GpuHost& gpu, int cycles) {
    SetStage("cycles-stress-init");
    TestWindow win;
    CHECK(win.Create(1280, 720, false));
    ComPtr<IDXGISwapChain4> sc;
    CHECK(gpu.CreateGameSwapChain(win.hwnd, 1280, 720, DXGI_FORMAT_R8G8B8A8_UNORM, &sc, 1));
    if (!sc) return;

    const ULONGLONG t0 = GetTickCount64();
    std::uint64_t hwfq_disp_total = 0;
    std::uint64_t dxgi_disp_total = 0;
    for (int c = 1; c <= cycles; ++c) {
        SetStage("cycle:hwfq-create");
        FgCompPresenterDesc cd;
        cd.device = gpu.device.Get();
        cd.hwnd = win.hwnd;
        cd.bind_window = true;
        cd.width = 1280;
        cd.height = 720;
        cd.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        cd.variable_refresh = true;
        cd.refresh_period = kRefresh360;
        cd.min_interval = FgVrrMinInterval(kRefresh360, false);
        cd.max_generated = 3;
        cd.slot_count = 12;
        cd.max_pending_intervals = 2;

        std::unique_ptr<FgWinCompPresenter> hp;
        const FgWinResult hcr = FgWinCompPresenter::Create(cd, &hp);
        CHECK(hcr.Ok() && hp);
        if (!hp) break;

        FgWinResult fw;
        const std::uint32_t hfid = hp->RegisterFence(gpu.fence.Get(), &fw);
        CHECK(hfid >= 1 && fw.Ok());
        CHECK(hp->Attach(0xC000 + static_cast<std::uint64_t>(c), 1) == FgStatus::Ok);

        std::atomic<std::uint64_t> h_retires{0};
        std::atomic<std::uint64_t> h_traces{0};
        hp->SetRealRetireCallback([](void* ctx, std::uint64_t, std::uint64_t, bool) {
            static_cast<std::atomic<std::uint64_t>*>(ctx)->fetch_add(1, std::memory_order_relaxed);
        }, &h_retires);
        hp->SetTraceCallback([](void* ctx, std::uint64_t, std::uint64_t, std::uint32_t, bool, std::uint8_t, FgTicks,
                                FgTicks) {
            static_cast<std::atomic<std::uint64_t>*>(ctx)->fetch_add(1, std::memory_order_relaxed);
        }, &h_traces);

        // Drive several real-only frames (leaving some in flight when Detach runs to stress teardown & handle race).
        SetStage("cycle:hwfq-present");
        FgTicks base = FgInterruptNow();
        for (std::uint64_t f = 1; f <= 6; ++f) {
            std::uint32_t slot = 0;
            if (hp->ReserveRealSlot(&slot) != FgStatus::Ok) break;
            const std::uint64_t fv = gpu.RenderClear(hp->SlotResource(slot), 0.1f, 0.3f, 0.7f);
            const FgTicks now = FgInterruptNow();
            const bool reanch = f > 1 && base < now;
            if (base < now) base = now;
            const FgIntervalPlan plan = MakePlan(1, f, f, base, cd.min_interval, 0, 2500, reanch);
            base = plan.targets[0].target;
            CHECK(hp->PresentRealOnly(plan, FgRealImage{1, f, FgReadyFence{hfid, fv}}) == FgStatus::Ok);
            SleepPump(4);
        }
        SleepPump(25);
        hwfq_disp_total += hp->Stats().displayed_real;

        // Teardown directly on the window thread (matching DestroySession on admission timeout / fallback).
        SetStage("cycle:hwfq-detach");
        FgStatus hds = hp->Detach();
        for (int i = 0; i < 100 && hds == FgStatus::Busy; ++i) {
            SleepPump(2);
            hds = hp->Detach();
        }
        CHECK(hds == FgStatus::Ok);
        SetStage("cycle:hwfq-destroy");
        hp.reset();
        CHECK(h_retires.load() >= 1 && h_traces.load() >= 1);
        gpu.WaitIdle();

        // Fallback to DXGI presenter on the same swap chain.
        SetStage("cycle:dxgi-create");
        FgDxgiPresenterDesc dd;
        dd.device = gpu.device.Get();
        dd.queue = gpu.present_queue.Get();
        dd.swap_chain = sc.Get();
        dd.pacing = gpu.tearing ? FgDxgiPacing::Tearing : FgDxgiPacing::Vsync;
        dd.variable_refresh = gpu.tearing;
        dd.refresh_period = kRefresh360;
        dd.min_interval = FgVrrMinInterval(kRefresh360, !gpu.tearing);
        dd.max_generated = 2;
        dd.slot_count = 10;
        std::unique_ptr<FgDxgiPresenter> dp;
        const FgWinResult dcr = FgDxgiPresenter::Create(dd, &dp);
        CHECK(dcr.Ok() && dp);
        if (!dp) break;

        const std::uint32_t dfid = dp->RegisterFence(gpu.fence.Get(), &fw);
        CHECK(dfid >= 1 && fw.Ok());
        CHECK(dp->Attach(0xD000 + static_cast<std::uint64_t>(c), 1) == FgStatus::Ok);

        SetStage("cycle:dxgi-present");
        FgTicks dbase = FgInterruptNow();
        for (std::uint64_t f = 1; f <= 3; ++f) {
            std::uint32_t slot = 0;
            if (dp->ReserveRealSlot(&slot) != FgStatus::Ok) break;
            const std::uint64_t fv = gpu.RenderClear(dp->SlotResource(slot), 0.6f, 0.2f, 0.2f);
            const FgTicks now = FgInterruptNow();
            const bool reanch = f > 1 && dbase < now;
            if (dbase < now) dbase = now;
            const FgIntervalPlan plan = MakePlan(1, f, f, dbase, dd.min_interval, 0, 2500, reanch);
            dbase = plan.targets[0].target;
            CHECK(dp->PresentRealOnly(plan, FgRealImage{1, f, FgReadyFence{dfid, fv}}) == FgStatus::Ok);
            SleepPump(4);
        }
        SleepPump(12);
        const FgPresentStats dst = dp->Stats();
        CHECK(dst.displayed_real >= 2 && dst.refused_calls == 0);
        dxgi_disp_total += dst.displayed_real;

        SetStage("cycle:dxgi-detach");
        CHECK(dp->Quiesce() == FgStatus::Ok);
        CHECK(dp->WaitIdle(1000));
        FgStatus dds = dp->Detach();
        for (int i = 0; i < 100 && dds == FgStatus::Busy; ++i) {
            SleepPump(2);
            dds = dp->Detach();
        }
        CHECK(dds == FgStatus::Ok);
        SetStage("cycle:dxgi-destroy");
        dp.reset();
        gpu.WaitIdle();
        PumpMessages();

        if (c == 1 || c % 10 == 0 || c == cycles) {
            std::printf("cycle %2d/%d ok (elapsed=%llums, hwfq_disp=%llu, dxgi_disp=%llu)\n", c, cycles,
                        static_cast<unsigned long long>(GetTickCount64() - t0),
                        static_cast<unsigned long long>(hwfq_disp_total),
                        static_cast<unsigned long long>(dxgi_disp_total));
            std::fflush(stdout);
        }
    }
    CHECK(hwfq_disp_total >= static_cast<std::uint64_t>(cycles));
    CHECK(dxgi_disp_total >= static_cast<std::uint64_t>(cycles) * 2);
}

void RunSmokeFallback(GpuHost& gpu, int cycles) {
    SetStage("smoke-init");
    TestWindow win;
    CHECK(win.Create(1280, 720, false, true, false));
    ComPtr<IDXGISwapChain4> sc;
    CHECK(gpu.CreateGameSwapChain(win.hwnd, 1280, 720, DXGI_FORMAT_R8G8B8A8_UNORM, &sc, 2));
    if (!sc) return;

    const ULONGLONG t0 = GetTickCount64();
    for (int c = 1; c <= cycles; ++c) {
        SetStage("smoke:hwfq-create");
        FgCompPresenterDesc cd;
        cd.device = gpu.device.Get();
        cd.hwnd = win.hwnd;
        cd.bind_window = true;
        cd.width = 1280;
        cd.height = 720;
        cd.format = DXGI_FORMAT_R8G8B8A8_UNORM;
        cd.variable_refresh = true;
        cd.refresh_period = kRefresh360;
        cd.min_interval = FgVrrMinInterval(kRefresh360, false);
        cd.max_generated = 3;
        cd.slot_count = 12;
        cd.max_pending_intervals = 2;

        std::unique_ptr<FgWinCompPresenter> hp;
        const FgWinResult hcr = FgWinCompPresenter::Create(cd, &hp);
        CHECK(hcr.Ok() && hp);
        if (!hp) break;
        const FgCompPresenterInfo init_info = hp->Info();
        CHECK(init_info.primed && init_info.hwnd_is_root && !init_info.window_cloaked);

        // Startup chain kick (matching TryCreateSession).
        {
            HANDLE w = sc->GetFrameLatencyWaitableObject();
            if (w) {
                WaitForSingleObject(w, 50);
                CloseHandle(w);
            }
            (void)sc->Present(0, 0);
            PumpMessages();
        }

        FgWinResult fw;
        const std::uint32_t hfid = hp->RegisterFence(gpu.fence.Get(), &fw);
        CHECK(hfid >= 1 && fw.Ok());
        CHECK(hp->Attach(0xE000 + static_cast<std::uint64_t>(c), 1) == FgStatus::Ok);

        std::atomic<std::uint64_t> h_retires{0};
        hp->SetRealRetireCallback([](void* ctx, std::uint64_t, std::uint64_t, bool) {
            static_cast<std::atomic<std::uint64_t>*>(ctx)->fetch_add(1, std::memory_order_relaxed);
        }, &h_retires);

        SetStage("smoke:hwfq-present");
        FgTicks base = FgInterruptNow();
        for (std::uint64_t f = 1; f <= 8; ++f) {
            std::uint32_t slot = 0;
            if (hp->ReserveRealSlot(&slot) != FgStatus::Ok) break;
            const std::uint64_t fv = gpu.RenderClear(hp->SlotResource(slot), 0.1f, 0.4f, 0.7f);
            const FgTicks now = FgInterruptNow();
            const bool reanch = f > 1 && base < now;
            if (base < now) base = now;
            const FgIntervalPlan plan = MakePlan(1, f, f, base, cd.min_interval, 0, 2500, reanch);
            base = plan.targets[0].target;
            CHECK(hp->PresentRealOnly(plan, FgRealImage{1, f, FgReadyFence{hfid, fv}}) == FgStatus::Ok);
            SleepPump(6);
        }
        SleepPump(35);
        const FgCompPresenterInfo ci = hp->Info();
        const FgPresentStats ps = hp->Stats();
        CHECK(ci.status_queued >= 1 && ci.status_frame_nonzero >= 1 && ci.status_target_count_max >= 1);

        SetStage("smoke:hwfq-detach");
        FgStatus hds = hp->Detach();
        for (int i = 0; i < 100 && hds == FgStatus::Busy; ++i) {
            SleepPump(2);
            hds = hp->Detach();
        }
        CHECK(hds == FgStatus::Ok);
        SetStage("smoke:hwfq-destroy");
        const ULONGLONG td0 = GetTickCount64();
        hp.reset();
        const ULONGLONG td_ms = GetTickCount64() - td0;
        CHECK(td_ms < 1000);
        gpu.WaitIdle();

        // Fallback to DXGI presenter on the same swap chain.
        SetStage("smoke:dxgi-fallback");
        FgDxgiPresenterDesc dd;
        dd.device = gpu.device.Get();
        dd.queue = gpu.present_queue.Get();
        dd.swap_chain = sc.Get();
        dd.pacing = gpu.tearing ? FgDxgiPacing::Tearing : FgDxgiPacing::Vsync;
        dd.variable_refresh = gpu.tearing;
        dd.refresh_period = kRefresh360;
        dd.min_interval = FgVrrMinInterval(kRefresh360, !gpu.tearing);
        dd.max_generated = 2;
        dd.slot_count = 10;
        std::unique_ptr<FgDxgiPresenter> dp;
        const FgWinResult dcr = FgDxgiPresenter::Create(dd, &dp);
        CHECK(dcr.Ok() && dp);
        if (!dp) break;
        const std::uint32_t dfid = dp->RegisterFence(gpu.fence.Get(), &fw);
        CHECK(dfid >= 1 && fw.Ok());
        CHECK(dp->Attach(0xF000 + static_cast<std::uint64_t>(c), 1) == FgStatus::Ok);
        FgTicks dbase = FgInterruptNow();
        for (std::uint64_t f = 1; f <= 3; ++f) {
            std::uint32_t slot = 0;
            if (dp->ReserveRealSlot(&slot) != FgStatus::Ok) break;
            const std::uint64_t fv = gpu.RenderClear(dp->SlotResource(slot), 0.7f, 0.2f, 0.2f);
            const FgTicks now = FgInterruptNow();
            const bool reanch = f > 1 && dbase < now;
            if (dbase < now) dbase = now;
            const FgIntervalPlan plan = MakePlan(1, f, f, dbase, dd.min_interval, 0, 2500, reanch);
            dbase = plan.targets[0].target;
            CHECK(dp->PresentRealOnly(plan, FgRealImage{1, f, FgReadyFence{dfid, fv}}) == FgStatus::Ok);
            SleepPump(4);
        }
        SleepPump(30);
        const FgPresentStats dst = dp->Stats();
        const FgDxgiPresenterInfo di = dp->Info();
        CHECK(di.presents >= 2 && dst.refused_calls == 0);
        CHECK(dp->Quiesce() == FgStatus::Ok && dp->WaitIdle(1000));
        FgStatus dds = dp->Detach();
        for (int i = 0; i < 100 && dds == FgStatus::Busy; ++i) {
            SleepPump(2);
            dds = dp->Detach();
        }
        CHECK(dds == FgStatus::Ok);
        dp.reset();
        gpu.WaitIdle();
        PumpMessages();
        std::printf("smoke cycle %d/%d ok: hwfq_sub=%llu q=%llu cfid_nz=%llu tgts=%u tct_nz=%llu rf=%llu wd=%llu disp=%llu td=%llums dxgi_pres=%llu dxgi_disp=%llu elapsed=%llums\n",
                    c, cycles, static_cast<unsigned long long>(ps.real_submitted),
                    static_cast<unsigned long long>(ci.status_queued),
                    static_cast<unsigned long long>(ci.status_frame_nonzero),
                    static_cast<unsigned>(ci.status_target_count_max),
                    static_cast<unsigned long long>(ci.status_target_completed_nonzero),
                    static_cast<unsigned long long>(ci.retiring_fence_value),
                    static_cast<unsigned long long>(ci.watchdog_retired),
                    static_cast<unsigned long long>(ps.displayed_real),
                    static_cast<unsigned long long>(td_ms),
                    static_cast<unsigned long long>(di.presents),
                    static_cast<unsigned long long>(dst.displayed_real),
                    static_cast<unsigned long long>(GetTickCount64() - t0));
    }
}

void RunR3Probe(GpuHost& gpu) {
    SetStage("r3probe");
    // Opt into Per-Monitor V2 DPI awareness so window/surface dimensions match physical monitor pixels 1:1.
    using SetDpiCtxFn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
    if (const HMODULE u32 = GetModuleHandleW(L"user32.dll")) {
        if (auto fn = reinterpret_cast<SetDpiCtxFn>(reinterpret_cast<void*>(
                GetProcAddress(u32, "SetProcessDpiAwarenessContext")))) {
            fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        }
    }
    const UINT scr_w = static_cast<UINT>(GetSystemMetrics(SM_CXSCREEN));
    const UINT scr_h = static_cast<UINT>(GetSystemMetrics(SM_CYSCREEN));
    const int nmon = GetSystemMetrics(SM_CMONITORS);
    DEVMODEW dm = {};
    dm.dmSize = sizeof(dm);
    EnumDisplaySettingsExW(nullptr, ENUM_CURRENT_SETTINGS, &dm, 0);
    std::printf("r3probe screen=%ux%u mode=%lux%lu@%luHz mons=%d\n", scr_w, scr_h,
                static_cast<unsigned long>(dm.dmPelsWidth), static_cast<unsigned long>(dm.dmPelsHeight),
                static_cast<unsigned long>(dm.dmDisplayFrequency), nmon);

    ComPtr<IDXGIOutput> out0;
    if (gpu.adapter && SUCCEEDED(gpu.adapter->EnumOutputs(0, &out0)) && out0) {
        ComPtr<IDXGIOutput6> out6;
        if (SUCCEEDED(out0.As(&out6))) {
            DXGI_OUTPUT_DESC1 od = {};
            if (SUCCEEDED(out6->GetDesc1(&od))) {
                std::printf("r3probe output0: rect=(%ld,%ld)-(%ld,%ld) cs=%d bpc=%u\n",
                            od.DesktopCoordinates.left, od.DesktopCoordinates.top,
                            od.DesktopCoordinates.right, od.DesktopCoordinates.bottom,
                            static_cast<int>(od.ColorSpace), static_cast<unsigned>(od.BitsPerColor));
            }
        }
        // Check overlay and overlay color-space support on a D3D11 device (as well as D3D12).
        const D3D_FEATURE_LEVEL qlevs[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        ComPtr<ID3D11Device> qd11;
        D3D11CreateDevice(gpu.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                          qlevs, 2, D3D11_SDK_VERSION, &qd11, nullptr, nullptr);
        ComPtr<IDXGIOutput4> out4;
        if (qd11 && SUCCEEDED(out0.As(&out4))) {
            UINT o_bgra = 0, o_rgba = 0, o_r10 = 0, o_fp16 = 0;
            out4->CheckOverlaySupport(DXGI_FORMAT_B8G8R8A8_UNORM, qd11.Get(), &o_bgra);
            out4->CheckOverlaySupport(DXGI_FORMAT_R8G8B8A8_UNORM, qd11.Get(), &o_rgba);
            out4->CheckOverlaySupport(DXGI_FORMAT_R10G10B10A2_UNORM, qd11.Get(), &o_r10);
            out4->CheckOverlaySupport(DXGI_FORMAT_R16G16B16A16_FLOAT, qd11.Get(), &o_fp16);
            UINT c_bgra = 0, c_r10_pq = 0, c_fp16_g10 = 0;
            out4->CheckOverlayColorSpaceSupport(DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
                                                qd11.Get(), &c_bgra);
            out4->CheckOverlayColorSpaceSupport(DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
                                                qd11.Get(), &c_r10_pq);
            out4->CheckOverlayColorSpaceSupport(DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,
                                                qd11.Get(), &c_fp16_g10);
            std::printf("r3probe d11_overlay: bgra=0x%x rgba=0x%x r10=0x%x fp16=0x%x | ov_cs: bgra_g22=0x%x r10_pq=0x%x fp16_g10=0x%x\n",
                        o_bgra, o_rgba, o_r10, o_fp16, c_bgra, c_r10_pq, c_fp16_g10);
        }
    }

    // 1) Baseline: DXGI FLIP_DISCARD fullscreen borderless with HDR10 PQ (matching output0 cs=12 bpc=10).
    for (int mode = 0; mode < 2; ++mode) {
        SetStage(mode == 0 ? "dxgi_hdr10_tear" : "dxgi_hdr10_sync1");
        TestWindow win;
        win.Create(scr_w, scr_h, false, false, false);
        ShowWindow(win.hwnd, SW_SHOW);
        SetForegroundWindow(win.hwnd);
        UpdateWindow(win.hwnd);
        SleepPump(50);
        const bool is_fg = GetForegroundWindow() == win.hwnd;
        ComPtr<IDXGISwapChain4> sc;
        if (gpu.CreateGameSwapChain(win.hwnd, scr_w, scr_h, DXGI_FORMAT_R10G10B10A2_UNORM, &sc, 0)) {
            const HRESULT cshr = sc->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020);
            COMPOSITION_FRAME_ID cf0 = 0, cf1 = 0;
            FgDCompGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &cf0);
            FgHighResTimer pt;
            FgTicks nxt = FgInterruptNow() + 41667;
            int presented = 0;
            for (int f = 0; f < 160; ++f) {
                if (mode == 0) {
                    if (FgInterruptNow() < nxt && pt.ArmAt(nxt)) WaitForSingleObject(pt.Handle(), 10);
                    nxt += 41667;
                }
                HANDLE w = sc->GetFrameLatencyWaitableObject();
                if (w) { WaitForSingleObject(w, 50); CloseHandle(w); }
                const UINT bi = sc->GetCurrentBackBufferIndex();
                ComPtr<ID3D12Resource> bb;
                if (SUCCEEDED(sc->GetBuffer(bi, IID_PPV_ARGS(&bb))) && bb) {
                    const std::uint64_t fv = gpu.RenderClear(bb.Get(), 0.0f, 0.7f, 0.0f,
                                                             D3D12_RESOURCE_STATE_PRESENT, gpu.present_queue.Get());
                    gpu.WaitFence(fv);
                }
                const UINT pflags = (mode == 0 && gpu.tearing) ? DXGI_PRESENT_ALLOW_TEARING : 0u;
                if (SUCCEEDED(sc->Present(mode == 0 ? 0u : 1u, pflags))) ++presented;
                PumpMessages();
            }
            FgDCompGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &cf1);
            DXGI_FRAME_STATISTICS fs = {};
            const HRESULT fhr = sc->GetFrameStatistics(&fs);
            std::printf("r5 %-16s: fg=%d cshr=0x%lx pres=%d cfid_d=%llu fstat_hr=0x%lx pres_cnt=%u sync_cnt=%u\n",
                        mode == 0 ? "dxgi_hdr10_tear" : "dxgi_hdr10_sync1", is_fg ? 1 : 0,
                        static_cast<unsigned long>(cshr), presented,
                        static_cast<unsigned long long>(cf1 >= cf0 ? cf1 - cf0 : 0),
                        static_cast<unsigned long>(fhr), fs.PresentCount, fs.SyncRefreshCount);
        }
        gpu.WaitIdle();
    }

    // 2) IPresentationManager with HDR10 PQ / FP16 scRGB matching output0 cs=12 bpc=10.
    struct R5Case {
        const char* name;
        DXGI_FORMAT fmt;
        DXGI_COLOR_SPACE_TYPE cs;
        bool set_is_hdr;
        bool dcomp_topmost;
        bool d12_shared;
        bool force_vsync;
        bool target_zero;
        bool set_pref_dur;
        FgTicks step_ticks;
        int frames;
    };
    const R5Case cases[] = {
        {"pm_hdr10_top0",   DXGI_FORMAT_R10G10B10A2_UNORM,  DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, true,  false, false, true,  false, false, 41667, 160},
        {"pm_hdr10_top1",   DXGI_FORMAT_R10G10B10A2_UNORM,  DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, true,  true,  false, true,  false, false, 41667, 160},
        {"pm_hdr10_d12",    DXGI_FORMAT_R10G10B10A2_UNORM,  DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, true,  false, true,  true,  false, false, 41667, 160},
        {"pm_hdr10_vs0",    DXGI_FORMAT_R10G10B10A2_UNORM,  DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, true,  false, false, false, false, false, 41667, 160},
        {"pm_hdr10_t0",     DXGI_FORMAT_R10G10B10A2_UNORM,  DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, true,  false, false, true,  true,  false, 83333,  80},
        {"pm_hdr10_pref",   DXGI_FORMAT_R10G10B10A2_UNORM,  DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, true,  false, false, true,  false, true,  41667, 160},
        {"pm_fp16_g10",     DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,    true,  false, false, true,  false, false, 41667, 160},
        {"pm_fp16_d12",     DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,    true,  true,  true,  true,  false, false, 41667, 160},
    };

    for (const R5Case& tc : cases) {
        SetStage(tc.name);
        const UINT vw = scr_w;
        const UINT vh = scr_h;
        TestWindow win;
        win.Create(vw, vh, false, false, false);
        ShowWindow(win.hwnd, SW_SHOW);
        SetForegroundWindow(win.hwnd);
        UpdateWindow(win.hwnd);
        SleepPump(40);
        const bool is_fg = GetForegroundWindow() == win.hwnd;

        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        const UINT flags11 = D3D11_CREATE_DEVICE_BGRA_SUPPORT |
                             D3D11_CREATE_DEVICE_PREVENT_INTERNAL_THREADING_OPTIMIZATIONS;
        ComPtr<ID3D11Device> d11r;
        ComPtr<ID3D11DeviceContext> c11r;
        HRESULT hr = D3D11CreateDevice(gpu.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags11, levels, 2,
                                       D3D11_SDK_VERSION, &d11r, nullptr, &c11r);
        ComPtr<ID3D11Device5> d11;
        ComPtr<ID3D11DeviceContext4> c11;
        if (SUCCEEDED(hr)) hr = d11r.As(&d11);
        if (SUCCEEDED(hr)) hr = c11r.As(&c11);
        ComPtr<IPresentationFactory> pfac;
        ComPtr<IPresentationManager> pmgr;
        if (SUCCEEDED(hr)) hr = FgDCompCreatePresentationFactory(d11.Get(), IID_PPV_ARGS(&pfac));
        if (SUCCEEDED(hr)) hr = pfac->CreatePresentationManager(&pmgr);

        constexpr int kBufs = 6;
        ComPtr<ID3D12Resource> d12_res[kBufs];
        ComPtr<ID3D11Texture2D> tex[kBufs];
        ComPtr<IPresentationBuffer> pbuf[kBufs];
        for (int i = 0; i < kBufs && SUCCEEDED(hr); ++i) {
            if (tc.d12_shared) {
                D3D12_HEAP_PROPERTIES hp = {};
                hp.Type = D3D12_HEAP_TYPE_DEFAULT;
                hp.CreationNodeMask = hp.VisibleNodeMask = 1;
                D3D12_RESOURCE_DESC rd = {};
                rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                rd.Width = vw;
                rd.Height = vh;
                rd.DepthOrArraySize = 1;
                rd.MipLevels = 1;
                rd.Format = tc.fmt;
                rd.SampleDesc.Count = 1;
                rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
                hr = gpu.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_DISPLAY,
                                                         &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&d12_res[i]));
                HANDLE h12 = nullptr;
                if (SUCCEEDED(hr)) hr = gpu.device->CreateSharedHandle(d12_res[i].Get(), nullptr, GENERIC_ALL, nullptr, &h12);
                if (SUCCEEDED(hr)) hr = d11->OpenSharedResource1(h12, IID_PPV_ARGS(&tex[i]));
                if (h12) CloseHandle(h12);
            } else {
                D3D11_TEXTURE2D_DESC td = {};
                td.Width = vw;
                td.Height = vh;
                td.MipLevels = 1;
                td.ArraySize = 1;
                td.Format = tc.fmt;
                td.SampleDesc.Count = 1;
                td.Usage = D3D11_USAGE_DEFAULT;
                td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
                td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                               D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE;
                hr = d11->CreateTexture2D(&td, nullptr, &tex[i]);
            }
            if (SUCCEEDED(hr)) hr = pmgr->AddBufferFromResource(tex[i].Get(), &pbuf[i]);
        }

        HANDLE sh = nullptr;
        ComPtr<IPresentationSurface> psurf;
        ComPtr<IDCompositionDesktopDevice> dcomp;
        ComPtr<IDCompositionTarget> dtarget;
        ComPtr<IDCompositionVisual2> dvis;
        ComPtr<IUnknown> dsurf;
        if (SUCCEEDED(hr)) hr = FgDCompCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, nullptr, &sh);
        if (SUCCEEDED(hr)) hr = pmgr->CreatePresentationSurface(sh, &psurf);
        if (SUCCEEDED(hr)) psurf->SetTag(77);
        if (SUCCEEDED(hr)) hr = psurf->SetAlphaMode(DXGI_ALPHA_MODE_IGNORE);
        if (SUCCEEDED(hr)) hr = psurf->SetColorSpace(tc.cs);
        if (SUCCEEDED(hr) && tc.set_is_hdr) {
            ComPtr<IPresentationSurface2> ps2;
            if (SUCCEEDED(psurf.As(&ps2)) && ps2) ps2->SetIsHdrContent(1);
        }
        const RECT src = {0, 0, static_cast<LONG>(vw), static_cast<LONG>(vh)};
        if (SUCCEEDED(hr)) hr = psurf->SetSourceRect(&src);

        if (SUCCEEDED(hr)) {
            if (tc.d12_shared) {
                const std::uint64_t fv = gpu.RenderClear(d12_res[0].Get(), 0.0f, 0.7f, 0.0f);
                gpu.WaitFence(fv);
            } else {
                ComPtr<ID3D11RenderTargetView> rtv;
                if (SUCCEEDED(d11->CreateRenderTargetView(tex[0].Get(), nullptr, &rtv))) {
                    const float g[4] = {0.0f, 0.7f, 0.0f, 1.0f};
                    c11->ClearRenderTargetView(rtv.Get(), g);
                    c11->Flush();
                }
            }
            hr = psurf->SetBuffer(pbuf[0].Get());
            if (SUCCEEDED(hr)) hr = pmgr->SetTargetTime(SystemInterruptTime{0});
            if (SUCCEEDED(hr)) hr = pmgr->Present();
        }
        ComPtr<IDXGIDevice> dxgi_dev;
        if (d11) d11.As(&dxgi_dev);
        if (SUCCEEDED(hr)) hr = FgDCompCreateDevice3(dxgi_dev.Get(), IID_PPV_ARGS(&dcomp));
        if (SUCCEEDED(hr)) hr = dcomp->CreateTargetForHwnd(win.hwnd, tc.dcomp_topmost ? TRUE : FALSE, &dtarget);
        if (SUCCEEDED(hr)) hr = dcomp->CreateVisual(&dvis);
        if (SUCCEEDED(hr)) hr = dcomp->CreateSurfaceFromHandle(sh, &dsurf);
        if (SUCCEEDED(hr)) hr = dvis->SetContent(dsurf.Get());
        if (SUCCEEDED(hr)) hr = dtarget->SetRoot(dvis.Get());
        if (SUCCEEDED(hr)) hr = dcomp->Commit();
        if (SUCCEEDED(hr)) dcomp->WaitForCommitCompletion();

        if (SUCCEEDED(hr)) hr = pmgr->ForceVSyncInterrupt(tc.force_vsync ? 1 : 0);
        if (SUCCEEDED(hr) && tc.set_pref_dur)
            (void)pmgr->SetPreferredPresentDuration(SystemInterruptTime{41667}, SystemInterruptTime{5000});
        if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_PresentStatus, 1);
        if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_CompositionFrame, 1);
        if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_IndependentFlipFrame, 1);

        ComPtr<ID3D11Fence> ret_fence;
        if (SUCCEEDED(hr)) pmgr->GetPresentRetiringFence(IID_PPV_ARGS(&ret_fence));
        const UINT64 rf0 = ret_fence ? ret_fence->GetCompletedValue() : 0;
        COMPOSITION_FRAME_ID cf0 = 0, cf1 = 0;
        FgDCompGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &cf0);

        int sub = 0, q = 0, sk = 0, ca = 0, ind = 0, cmp = 0;
        std::vector<FgTicks> ind_err_us, ind_dt_us;
        std::vector<std::pair<UINT64, FgTicks>> pid_tgt;
        FgTicks last_ind_t = 0;
        auto drain = [&]() {
            for (int g = 0; g < 256; ++g) {
                ComPtr<IPresentStatistics> st;
                if (!pmgr || pmgr->GetNextPresentStatistics(&st) != S_OK || !st) break;
                const PresentStatisticsKind k = st->GetKind();
                const UINT64 pid = st->GetPresentId();
                if (k == PresentStatisticsKind_PresentStatus) {
                    ComPtr<IPresentStatusPresentStatistics> ps;
                    if (SUCCEEDED(st.As(&ps))) {
                        const PresentStatus pv = ps->GetPresentStatus();
                        if (pv == PresentStatus_Queued) ++q;
                        else if (pv == PresentStatus_Skipped) ++sk;
                        else if (pv == PresentStatus_Canceled) ++ca;
                    }
                } else if (k == PresentStatisticsKind_IndependentFlipFrame) {
                    ++ind;
                    ComPtr<IIndependentFlipFramePresentStatistics> fs;
                    if (SUCCEEDED(st.As(&fs))) {
                        const FgTicks dt = static_cast<FgTicks>(fs->GetDisplayedTime().value);
                        if (dt > 0) {
                            if (last_ind_t > 0 && dt > last_ind_t) ind_dt_us.push_back((dt - last_ind_t) / 10);
                            last_ind_t = dt;
                            for (const auto& pt : pid_tgt) {
                                if (pt.first == pid && pt.second > 0) {
                                    ind_err_us.push_back((dt - pt.second) / 10);
                                    break;
                                }
                            }
                        }
                    }
                } else if (k == PresentStatisticsKind_CompositionFrame) {
                    ++cmp;
                }
            }
        };

        FgHighResTimer pace_timer;
        FgTicks next_target = FgInterruptNow() + 60000;
        for (int f = 0; f < tc.frames && SUCCEEDED(hr); ++f) {
            const int s = (f + 1) % kBufs;
            const FgTicks wake_at = tc.target_zero ? next_target : (next_target - 20000);
            if (FgInterruptNow() < wake_at && pace_timer.ArmAt(wake_at))
                WaitForSingleObject(pace_timer.Handle(), 20);
            PumpMessages();
            drain();
            const ULONGLONG w_until = GetTickCount64() + 40;
            boolean is_av = 0;
            while (GetTickCount64() < w_until) {
                if (SUCCEEDED(pbuf[s]->IsAvailable(&is_av)) && is_av) break;
                drain();
                Sleep(1);
            }
            if (!is_av) break;
            if (tc.d12_shared) {
                const std::uint64_t fv = gpu.RenderClear(d12_res[s].Get(), 0.0f, 0.7f, 0.0f);
                gpu.WaitFence(fv);
            } else {
                ComPtr<ID3D11RenderTargetView> rtv;
                if (SUCCEEDED(d11->CreateRenderTargetView(tex[s].Get(), nullptr, &rtv))) {
                    const float g[4] = {0.0f, 0.7f, 0.0f, 1.0f};
                    c11->ClearRenderTargetView(rtv.Get(), g);
                    c11->Flush();
                }
            }
            const UINT64 pid = pmgr->GetNextPresentId();
            const FgTicks req_t = tc.target_zero ? 0 : next_target;
            hr = psurf->SetBuffer(pbuf[s].Get());
            if (SUCCEEDED(hr)) hr = pmgr->SetTargetTime(SystemInterruptTime{static_cast<UINT64>(req_t)});
            if (SUCCEEDED(hr)) hr = pmgr->Present();
            if (SUCCEEDED(hr)) {
                ++sub;
                pid_tgt.push_back({pid, req_t});
            }
            next_target += tc.step_ticks;
        }
        SleepPump(60);
        drain();
        FgDCompGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &cf1);
        const UINT64 rf1 = ret_fence ? ret_fence->GetCompletedValue() : 0;
        COLORREF px = 0xFFFFFFFFu;
        {
            HDC sdc = GetDC(nullptr);
            if (sdc) { px = GetPixel(sdc, 200, 200); ReleaseDC(nullptr, sdc); }
        }
        auto pct = [](std::vector<FgTicks>& v, int p) -> long long {
            if (v.empty()) return 0;
            std::sort(v.begin(), v.end());
            const std::size_t idx = (v.size() - 1) * static_cast<std::size_t>(p) / 100;
            return static_cast<long long>(v[idx]);
        };

        if (dvis) dvis->SetContent(nullptr);
        if (dtarget) dtarget->SetRoot(nullptr);
        if (dcomp) dcomp->Commit();
        dsurf.Reset(); dvis.Reset(); dtarget.Reset(); dcomp.Reset(); psurf.Reset();
        for (int i = 0; i < kBufs; ++i) { pbuf[i].Reset(); tex[i].Reset(); d12_res[i].Reset(); }
        ret_fence.Reset(); pmgr.Reset(); pfac.Reset(); c11.Reset(); d11.Reset(); c11r.Reset(); d11r.Reset();
        if (sh) CloseHandle(sh);

        std::printf("r5 %-16s: hr=0x%lx fg=%d px=0x%06lx sub=%3d q=%3d sk=%3d ind=%3d(dt=%lldus err=%lld/%lld/%lldus) cmp=%2d rf=%3llu cfid_d=%3llu\n",
                    tc.name, static_cast<unsigned long>(hr), is_fg ? 1 : 0, static_cast<unsigned long>(px),
                    sub, q, sk, ind, pct(ind_dt_us, 50), pct(ind_err_us, 50), pct(ind_err_us, 95), pct(ind_err_us, 100),
                    cmp, static_cast<unsigned long long>(rf1 - rf0),
                    static_cast<unsigned long long>(cf1 >= cf0 ? cf1 - cf0 : 0));
    }
}

void RunPhaseProbe(GpuHost& gpu) {
    SetStage("phaseprobe");
    EnablePerMonitorDpiV2();
    const UINT scr_w = static_cast<UINT>(GetSystemMetrics(SM_CXSCREEN));
    const UINT scr_h = static_cast<UINT>(GetSystemMetrics(SM_CYSCREEN));
    TestWindow win;
    win.Create(scr_w, scr_h, false, false, false);
    ShowWindow(win.hwnd, SW_SHOW);
    SetForegroundWindow(win.hwnd);
    UpdateWindow(win.hwnd);
    SleepPump(40);

    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    const UINT flags11 = D3D11_CREATE_DEVICE_BGRA_SUPPORT |
                         D3D11_CREATE_DEVICE_PREVENT_INTERNAL_THREADING_OPTIMIZATIONS;
    ComPtr<ID3D11Device> d11r;
    ComPtr<ID3D11DeviceContext> c11r;
    HRESULT hr = D3D11CreateDevice(gpu.adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags11, levels, 2,
                                   D3D11_SDK_VERSION, &d11r, nullptr, &c11r);
    ComPtr<ID3D11Device5> d11;
    ComPtr<ID3D11DeviceContext4> c11;
    if (SUCCEEDED(hr)) hr = d11r.As(&d11);
    if (SUCCEEDED(hr)) hr = c11r.As(&c11);
    ComPtr<IPresentationFactory> pfac;
    ComPtr<IPresentationManager> pmgr;
    if (SUCCEEDED(hr)) hr = FgDCompCreatePresentationFactory(d11.Get(), IID_PPV_ARGS(&pfac));
    if (SUCCEEDED(hr)) hr = pfac->CreatePresentationManager(&pmgr);

    constexpr int kBufs = 6;
    ComPtr<ID3D11Texture2D> tex[kBufs];
    ComPtr<IPresentationBuffer> pbuf[kBufs];
    for (int i = 0; i < kBufs && SUCCEEDED(hr); ++i) {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = scr_w;
        td.Height = scr_h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                       D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE;
        hr = d11->CreateTexture2D(&td, nullptr, &tex[i]);
        if (SUCCEEDED(hr)) {
            ComPtr<ID3D11RenderTargetView> rtv;
            if (SUCCEEDED(d11->CreateRenderTargetView(tex[i].Get(), nullptr, &rtv))) {
                const float g[4] = {0.0f, 1.0f, 0.0f, 1.0f};
                c11->ClearRenderTargetView(rtv.Get(), g);
            }
            hr = pmgr->AddBufferFromResource(tex[i].Get(), &pbuf[i]);
        }
    }
    if (c11) c11->Flush();

    HANDLE sh = nullptr;
    ComPtr<IPresentationSurface> psurf;
    ComPtr<IDCompositionDesktopDevice> dcomp;
    ComPtr<IDCompositionTarget> dtarget;
    ComPtr<IDCompositionVisual2> dvis;
    ComPtr<IUnknown> dsurf;
    if (SUCCEEDED(hr)) hr = FgDCompCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, nullptr, &sh);
    if (SUCCEEDED(hr)) hr = pmgr->CreatePresentationSurface(sh, &psurf);
    if (SUCCEEDED(hr)) psurf->SetTag(77);
    if (SUCCEEDED(hr)) hr = psurf->SetAlphaMode(DXGI_ALPHA_MODE_IGNORE);
    if (SUCCEEDED(hr)) hr = psurf->SetColorSpace(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
    const RECT src = {0, 0, static_cast<LONG>(scr_w), static_cast<LONG>(scr_h)};
    if (SUCCEEDED(hr)) hr = psurf->SetSourceRect(&src);
    if (SUCCEEDED(hr)) {
        psurf->SetBuffer(pbuf[0].Get());
        pmgr->SetTargetTime(SystemInterruptTime{0});
        pmgr->Present();
    }
    ComPtr<IDXGIDevice> dxgi_dev;
    if (d11) d11.As(&dxgi_dev);
    if (SUCCEEDED(hr)) hr = FgDCompCreateDevice3(dxgi_dev.Get(), IID_PPV_ARGS(&dcomp));
    if (SUCCEEDED(hr)) hr = dcomp->CreateTargetForHwnd(win.hwnd, TRUE, &dtarget);
    if (SUCCEEDED(hr)) hr = dcomp->CreateVisual(&dvis);
    if (SUCCEEDED(hr)) hr = dcomp->CreateSurfaceFromHandle(sh, &dsurf);
    if (SUCCEEDED(hr)) hr = dvis->SetContent(dsurf.Get());
    if (SUCCEEDED(hr)) hr = dtarget->SetRoot(dvis.Get());
    if (SUCCEEDED(hr)) hr = dcomp->Commit();
    if (SUCCEEDED(hr)) dcomp->WaitForCommitCompletion();
    if (SUCCEEDED(hr)) hr = pmgr->ForceVSyncInterrupt(1);
    if (SUCCEEDED(hr)) hr = pmgr->EnablePresentStatisticsKind(PresentStatisticsKind_PresentStatus, 1);

    FgClockMap clk;
    FgRefreshGrid grid;
    CHECK(FgReadCompositorGrid(clk, 0, &grid));
    // Also inspect startTime vs targetTime vs presentedStats.time on a completed frame.
    {
        COMPOSITION_FRAME_ID cid = 0;
        FgDCompGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &cid);
        COMPOSITION_FRAME_STATS fst = {};
        COMPOSITION_TARGET_ID tgts[4] = {};
        UINT cnt = 0;
        COMPOSITION_TARGET_STATS ts = {};
        if (SUCCEEDED(FgDCompGetStatistics(cid, &fst, 4, tgts, &cnt)) && cnt > 0 &&
            SUCCEEDED(FgDCompGetTargetStatistics(cid, &tgts[0], &ts))) {
            const FgTicks st_it = clk.FromQpc(static_cast<std::int64_t>(fst.startTime));
            const FgTicks tg_it = clk.FromQpc(static_cast<std::int64_t>(fst.targetTime));
            const FgTicks pr_it = ts.presentedStats.time ? clk.FromQpc(static_cast<std::int64_t>(ts.presentedStats.time)) : 0;
            std::printf("dcomp_frame: cid=%llu period=%lldus start-target=%lldus pres-target=%lldus vblankDur=%llu\n",
                        static_cast<unsigned long long>(cid), static_cast<long long>(grid.period / 10),
                        static_cast<long long>((st_it - tg_it) / 10),
                        static_cast<long long>(pr_it ? (pr_it - tg_it) / 10 : -999999),
                        static_cast<unsigned long long>(ts.vblankDuration));
        }
    }

    // Drain prime stat.
    SleepPump(20);
    {
        ComPtr<IPresentStatistics> st;
        while (pmgr->GetNextPresentStatistics(&st) == S_OK && st) st.Reset();
    }

    // 1) Sweep 16 submit phases relative to grid.origin + k*period (spaced 4 periods apart so no interference).
    FgHighResTimer timer;
    const FgTicks P = grid.period;
    for (int use_future_target = 0; use_future_target < 2; ++use_future_target) {
        std::printf("--- phase sweep (future_target=%s) ---\n", use_future_target ? "+3P" : "0");
        for (int ph = 0; ph < 16; ++ph) {
            const FgTicks now = FgInterruptNow();
            const FgTicks k = (now - grid.origin) / P + 4;
            const FgTicks vk = grid.origin + k * P;
            const FgTicks wake_at = vk + (ph * P) / 16;
            if (FgInterruptNow() < wake_at && timer.ArmAt(wake_at)) WaitForSingleObject(timer.Handle(), 20);
            const FgTicks t_sub = FgInterruptNow();
            const int s = ph % kBufs;
            const UINT64 pid = pmgr->GetNextPresentId();
            const FgTicks req_t = use_future_target ? (vk + 3 * P) : 0;
            psurf->SetBuffer(pbuf[s].Get());
            pmgr->SetTargetTime(SystemInterruptTime{static_cast<UINT64>(req_t)});
            pmgr->Present();

            COMPOSITION_FRAME_ID cfid = 0;
            int pv_got = -1;
            const ULONGLONG until = GetTickCount64() + 50;
            while (GetTickCount64() < until && cfid == 0) {
                ComPtr<IPresentStatistics> st;
                while (pmgr->GetNextPresentStatistics(&st) == S_OK && st) {
                    if (st->GetPresentId() == pid && st->GetKind() == PresentStatisticsKind_PresentStatus) {
                        ComPtr<IPresentStatusPresentStatistics> ps;
                        if (SUCCEEDED(st.As(&ps))) {
                            cfid = ps->GetCompositionFrameId();
                            pv_got = static_cast<int>(ps->GetPresentStatus());
                        }
                    }
                    st.Reset();
                }
                if (cfid == 0) Sleep(1);
            }
            // Wait for cfid completion.
            COMPOSITION_FRAME_ID comp_id = 0;
            while (GetTickCount64() < until + 40) {
                FgDCompGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &comp_id);
                if (comp_id >= cfid && cfid != 0) break;
                Sleep(1);
            }
            FgTicks pres_t = 0, cf_tgt = 0, cf_start = 0;
            if (cfid != 0) {
                COMPOSITION_FRAME_STATS fst = {};
                COMPOSITION_TARGET_ID tgts[4] = {};
                UINT cnt = 0;
                COMPOSITION_TARGET_STATS ts = {};
                if (SUCCEEDED(FgDCompGetStatistics(cfid, &fst, 4, tgts, &cnt))) {
                    cf_start = clk.FromQpc(static_cast<std::int64_t>(fst.startTime));
                    cf_tgt = clk.FromQpc(static_cast<std::int64_t>(fst.targetTime));
                    if (cnt > 0 && SUCCEEDED(FgDCompGetTargetStatistics(cfid, &tgts[0], &ts)) && ts.presentedStats.time)
                        pres_t = clk.FromQpc(static_cast<std::int64_t>(ts.presentedStats.time));
                }
            }
            std::printf("ph=%2d/16 sub-Vk=%4lldus | st=%d cf_start-Vk=%5lldus cf_tgt-Vk=%5lldus pres-Vk=%5lldus (pres-sub=%4lldus)\n",
                        ph, static_cast<long long>((t_sub - vk) / 10), pv_got,
                        static_cast<long long>(cf_start ? (cf_start - vk) / 10 : -9999),
                        static_cast<long long>(cf_tgt ? (cf_tgt - vk) / 10 : -9999),
                        static_cast<long long>(pres_t ? (pres_t - vk) / 10 : -9999),
                        static_cast<long long>(pres_t ? (pres_t - t_sub) / 10 : -9999));
        }
    }

    // 2) Full 360 Hz 1-per-vblank test at 4 different submit phases (0, 25, 50, 75% of P).
    for (int qph = 0; qph < 4; ++qph) {
        SetStage("phase360");
        FgReadCompositorGrid(clk, 0, &grid);
        const FgTicks now = FgInterruptNow();
        FgTicks vk = grid.origin + ((now - grid.origin) / P + 4) * P;
        const FgTicks offset = (qph * P) / 4;
        int sub = 0, q = 0, sk = 0, short_dt = 0;
        FgTicks last_pres = 0;
        std::vector<COMPOSITION_FRAME_ID> q_cfids;
        for (int f = 0; f < 300; ++f) {
            const FgTicks wake_at = vk + offset;
            if (FgInterruptNow() < wake_at && timer.ArmAt(wake_at)) WaitForSingleObject(timer.Handle(), 15);
            const int s = f % kBufs;
            psurf->SetBuffer(pbuf[s].Get());
            pmgr->SetTargetTime(SystemInterruptTime{0});
            if (SUCCEEDED(pmgr->Present())) ++sub;
            ComPtr<IPresentStatistics> st;
            while (pmgr->GetNextPresentStatistics(&st) == S_OK && st) {
                if (st->GetKind() == PresentStatisticsKind_PresentStatus) {
                    ComPtr<IPresentStatusPresentStatistics> ps;
                    if (SUCCEEDED(st.As(&ps))) {
                        if (ps->GetPresentStatus() == PresentStatus_Queued) {
                            ++q;
                            q_cfids.push_back(ps->GetCompositionFrameId());
                        } else if (ps->GetPresentStatus() == PresentStatus_Skipped) ++sk;
                    }
                }
                st.Reset();
            }
            vk += P;
        }
        SleepPump(30);
        {
            ComPtr<IPresentStatistics> st;
            while (pmgr->GetNextPresentStatistics(&st) == S_OK && st) {
                if (st->GetKind() == PresentStatisticsKind_PresentStatus) {
                    ComPtr<IPresentStatusPresentStatistics> ps;
                    if (SUCCEEDED(st.As(&ps))) {
                        if (ps->GetPresentStatus() == PresentStatus_Queued) {
                            ++q;
                            q_cfids.push_back(ps->GetCompositionFrameId());
                        } else if (ps->GetPresentStatus() == PresentStatus_Skipped) ++sk;
                    }
                }
                st.Reset();
            }
        }
        int same_cfid = 0, same_pres = 0;
        COMPOSITION_FRAME_ID prev_cf = 0;
        for (COMPOSITION_FRAME_ID cf : q_cfids) {
            if (cf == prev_cf) ++same_cfid;
            prev_cf = cf;
            COMPOSITION_FRAME_STATS fst = {};
            COMPOSITION_TARGET_ID tgts[4] = {};
            UINT cnt = 0;
            COMPOSITION_TARGET_STATS ts = {};
            if (SUCCEEDED(FgDCompGetStatistics(cf, &fst, 4, tgts, &cnt)) && cnt > 0 &&
                SUCCEEDED(FgDCompGetTargetStatistics(cf, &tgts[0], &ts)) && ts.presentedStats.time) {
                const FgTicks pt = clk.FromQpc(static_cast<std::int64_t>(ts.presentedStats.time));
                if (last_pres > 0) {
                    if (pt == last_pres) ++same_pres;
                    if (pt - last_pres < P - P / 10) ++short_dt;
                }
                last_pres = pt;
            }
        }
        std::printf("360hz phase=%d/4 (%4lldus): sub=%d q=%d sk=%d (%.2f%%) same_cfid=%d same_pres=%d short=%d\n",
                    qph, static_cast<long long>(offset / 10), sub, q, sk,
                    sub > 0 ? 100.0 * sk / sub : 0.0, same_cfid, same_pres, short_dt);
    }

    if (dvis) dvis->SetContent(nullptr);
    if (dtarget) dtarget->SetRoot(nullptr);
    if (dcomp) dcomp->Commit();
    dsurf.Reset(); dvis.Reset(); dtarget.Reset(); dcomp.Reset(); psurf.Reset();
    for (int i = 0; i < kBufs; ++i) { pbuf[i].Reset(); tex[i].Reset(); }
    pmgr.Reset(); pfac.Reset(); c11.Reset(); d11.Reset(); c11r.Reset(); d11r.Reset();
    if (sh) CloseHandle(sh);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool probe = false;
    bool diag = false;
    bool smoke = false;
    bool stress = false;
    bool gate = false;
    bool r3probe = false;
    bool phaseprobe = false;
    int cycles = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--probe") == 0) probe = true;
        else if (std::strcmp(argv[i], "--diag") == 0) diag = true;
        else if (std::strcmp(argv[i], "--smoke") == 0) smoke = true;
        else if (std::strcmp(argv[i], "--stress") == 0) stress = true;
        else if (std::strcmp(argv[i], "--gate") == 0) gate = true;
        else if (std::strcmp(argv[i], "--r3probe") == 0) r3probe = true;
        else if (std::strcmp(argv[i], "--phaseprobe") == 0) phaseprobe = true;
        else if (std::strcmp(argv[i], "--cycles") == 0 && i + 1 < argc) cycles = std::atoi(argv[++i]);
    }
    if (!probe && !diag && !smoke && !stress && !gate && !r3probe && !phaseprobe) smoke = true;

    SetStage("init");
    std::thread watchdog([]() {
        while (!g_watchdog_stop.load(std::memory_order_acquire)) {
            Sleep(250);
            const ULONGLONG since = g_stage_since.load(std::memory_order_acquire);
            const ULONGLONG age = since ? (GetTickCount64() - since) : 0;
            if (age > 4000) {
                std::printf("[WATCHDOG] stage '%s' stalled for %llu ms! Exiting.\n",
                            g_stage.load(std::memory_order_acquire), static_cast<unsigned long long>(age));
                std::fflush(stdout);
                ExitProcess(99);
            }
        }
    });

    GpuHost gpu;
    CHECK(gpu.Init());
    if (gpu.device) {
        if (r3probe) {
            RunR3Probe(gpu);
        }
        if (phaseprobe) {
            RunPhaseProbe(gpu);
        }
        if (diag) {
            RunDiag(gpu);
        }
        if (smoke) {
            RunSmokeFallback(gpu, cycles);
        }
        if (probe) {
            const UINT scr_w = static_cast<UINT>(GetSystemMetrics(SM_CXSCREEN));
            const UINT scr_h = static_cast<UINT>(GetSystemMetrics(SM_CYSCREEN));
            const RawCellOpts cells[] = {
                {"no_sc,topwin,dcomp(dxgi),top=1,t+8ms", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, false, 0, false, true, true, true, false, false, false, true, false, 80000, true, false, true, false, false},
                {"no_sc,topwin,dcomp(null),top=1,t+8ms", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, false, 0, false, false, true, true, false, false, false, true, false, 80000, true, false, true, false, false},
                {"no_sc,topwin,pure11,dcomp(dxgi),t+8ms", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, false, 0, false, true, true, true, true, false, true, true, false, 80000, true, false, true, false, false},
                {"no_sc,topwin,noredir,dcomp(dxgi),t+8ms", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, false, 0, false, true, true, true, false, false, false, true, false, 80000, true, false, true, true, false},
                {"no_sc,topwin,fullscreen,dcomp(dxgi)", scr_w, scr_h, DXGI_FORMAT_R8G8B8A8_UNORM, false, 0, false, true, true, true, false, false, false, true, false, 80000, true, false, true, false, false},
                {"sc_1p,topwin,dcomp(dxgi),top=1,t+8ms", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, true, 1, false, true, true, true, false, false, false, true, false, 80000, true, false, true, false, false},
                {"sc_1p,topwin,dcomp(dxgi),top=0,t+8ms", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, true, 1, false, true, false, true, false, false, false, true, false, 80000, true, false, true, false, false},
                {"sc_1p,topwin,child_hwnd,dcomp(dxgi)", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, true, 1, true, true, true, true, false, false, false, true, false, 80000, true, false, true, false, false},
                {"sc_interl,topwin,dcomp(dxgi),top=1", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, true, 1, false, true, true, true, false, false, false, true, false, 80000, true, false, true, false, true},
                {"sc_interl,topwin,child_hwnd,dcomp(dxgi)", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, true, 1, true, true, true, true, false, false, false, true, false, 80000, true, false, true, false, true},
                {"sc_1p,topwin,dcomp(dxgi),t=0", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, true, 1, false, true, true, true, false, false, false, true, true, 0, true, false, true, false, false},
                {"sc_1p,topwin,dcomp(dxgi),BGRA,flush11", 1920, 1080, DXGI_FORMAT_B8G8R8A8_UNORM, true, 1, false, true, true, true, true, false, false, true, false, 28333, true, false, true, false, false},
                {"sc_1p,topwin,dcomp(dxgi),detach_pres+wait", 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM, true, 1, false, true, true, true, false, false, false, true, false, 80000, true, true, true, false, false},
            };
            for (const RawCellOpts& c : cells) RunRawProbeCell(gpu, c);
            RunPresenterProbe(gpu);
        }
        if (gate) {
            RunJitteredBaseGate(gpu);
            RunFullscreenPaced240Gate(gpu);
            RunLongHwfqRun(gpu);
            RunCyclesStress(gpu, cycles);
        }
        if (stress) {
            RunLongHwfqRun(gpu);
            RunCyclesStress(gpu, cycles);
        }
    }

    g_watchdog_stop.store(true, std::memory_order_release);
    if (watchdog.joinable()) watchdog.join();
    std::printf("test_fg_hwfq_window: %d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
