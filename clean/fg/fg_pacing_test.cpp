// SPDX-License-Identifier: MIT
// fg_pacing_test: presents alternating "real" and "stand-in" frames at a fixed target interval and measures how evenly
// they reach the screen, through two public Windows presentation routes:
//
//   route A  composition swapchain (presentation.h): one present per frame with an app-chosen target time
//            (IPresentationManager::SetTargetTime). On-screen times come from the API's present statistics:
//            IIndependentFlipFramePresentStatistics::GetDisplayedTime when the present was independent-flipped, and
//            for composed presents the composition frame's present time (DCompositionGetStatistics /
//            DCompositionGetTargetStatistics). Displayable surfaces are a Direct3D 11 feature, so this route renders
//            with a D3D11 device.
//   route B  DXGI flip model on Direct3D 12 (IDXGISwapChain4, DXGI_SWAP_EFFECT_FLIP_DISCARD, frame-latency waitable
//            object, Present(1, 0)). On-screen times come from IDXGISwapChain::GetFrameStatistics.
//
// The two frame kinds are two low-contrast greys (no strobing). Nothing is installed or changed; the program only
// opens a window for the run. Output: one summary block on stdout and a CSV of per-present times (--out).
// Written for this project from Microsoft's public API documentation; no third-party code.
//
// usage: fg_pacing_test --route A|B [--mode window|borderless] [--seconds S] [--interval-refreshes N] [--out file.csv]
//                       [--timebase it|qpc] [--force-vsync-interrupt] [--verbose]
// It also prints read-only context first: display mode and refresh period, the output's EDID identity, tearing
// support, the console display state, and DWM's completed frames over one second. On-screen times need a live
// monitor: without one (no EDID, no scanout VSync) the APIs return no display times and the summary says so.
#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <dcomp.h>
#include <presentation.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

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
    char route = 'A';
    bool borderless = false;
    double seconds = 5.0;
    int interval_refreshes = 1;
    bool qpc_targets = false;  // --timebase qpc: pass QPC values to SetTargetTime instead of interrupt time
    bool force_vsync = false;  // --force-vsync-interrupt: IPresentationManager::ForceVSyncInterrupt(true)
    bool verbose = false;      // --verbose: the first presents' raw statistics and waits
    std::string out;
};

struct Sample {             // one present
    UINT64 id = 0;
    int kind = 0;           // 0 real, 1 stand-in
    double target_ms = 0;   // requested time (route A) or present call time (route B), ms on the run's clock
    double shown_ms = -1;   // on-screen time from the API, ms on the same clock; -1 unknown
    std::string how;        // "iflip", "composed", "skipped", "canceled", "dxgi"
};

double g_qpc_ms = 0;        // ms per QPC tick
UINT64 g_t0_qpc = 0;
ULONGLONG g_t0_it = 0;      // interrupt time (100 ns) at the same moment

double qpc_to_ms(UINT64 q) { return (static_cast<double>(q) - static_cast<double>(g_t0_qpc)) * g_qpc_ms; }
double it_to_ms(ULONGLONG t) { return (static_cast<double>(t) - static_cast<double>(g_t0_it)) / 10000.0; }

void init_clocks() {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    g_qpc_ms = 1000.0 / static_cast<double>(f.QuadPart);
    ULONGLONG it;
    QueryInterruptTimePrecise(&it);
    QueryPerformanceCounter(&c);
    g_t0_it = it;
    g_t0_qpc = static_cast<UINT64>(c.QuadPart);
}

bool g_quit = false;
int g_display_state = -1;  // GUID_CONSOLE_DISPLAY_STATE: 0 off, 1 on, 2 dimmed; -1 not reported
LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_POWERBROADCAST && w == PBT_POWERSETTINGCHANGE) {
        const POWERBROADCAST_SETTING *ps = reinterpret_cast<const POWERBROADCAST_SETTING *>(l);
        if (ps && IsEqualGUID(ps->PowerSetting, GUID_CONSOLE_DISPLAY_STATE) && ps->DataLength >= sizeof(DWORD))
            g_display_state = static_cast<int>(*reinterpret_cast<const DWORD *>(ps->Data));
        return TRUE;
    }
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

HWND make_window(bool borderless, bool no_redirection, RECT *client) {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"wg_fg_pacing_test";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    HMONITOR mon = MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO mi = {sizeof(mi)};
    GetMonitorInfoW(mon, &mi);
    // Topmost: the program is started from a background process and may not get the foreground; an occluded window's
    // presents are throttled by the system, which would measure the occlusion, not the presentation route.
    DWORD ex = (no_redirection ? WS_EX_NOREDIRECTIONBITMAP : 0) | WS_EX_TOPMOST;
    RECT r = mi.rcMonitor;
    DWORD style = WS_POPUP;
    if (!borderless) {
        style = WS_OVERLAPPEDWINDOW;
        r = RECT{mi.rcWork.left + 64, mi.rcWork.top + 64, mi.rcWork.left + 64 + 1280, mi.rcWork.top + 64 + 720};
        AdjustWindowRectEx(&r, style, FALSE, ex);
    }
    HWND h = CreateWindowExW(ex, wc.lpszClassName, L"WrongGPU FG pacing test (Esc ends)", style, r.left, r.top,
                             r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
    ShowWindow(h, SW_SHOW);
    UpdateWindow(h);
    GetClientRect(h, client);
    return h;
}

ComPtr<IDXGIAdapter1> pick_adapter() {
    ComPtr<IDXGIFactory6> f;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&f));
    ComPtr<IDXGIAdapter1> a;
    for (UINT i = 0; f->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&a)) !=
                     DXGI_ERROR_NOT_FOUND;
         ++i) {
        DXGI_ADAPTER_DESC1 d;
        a->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) return a;
    }
    return nullptr;
}

// Read-only context: display mode, tearing support, and the user's DirectX graphics preferences string.
double print_context(HWND hwnd, const Options &o) {
    HMONITOR mon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFOEXW mi = {};
    mi.cbSize = sizeof(mi);
    GetMonitorInfoW(mon, &mi);
    DEVMODEW dm = {};
    dm.dmSize = sizeof(dm);
    EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm);
    std::printf("display: %ld x %ld, %lu Hz (EnumDisplaySettings), window mode %s\n",
                static_cast<long>(dm.dmPelsWidth), static_cast<long>(dm.dmPelsHeight),
                static_cast<unsigned long>(dm.dmDisplayFrequency), o.borderless ? "borderless covering the display" : "windowed 1280x720");
    // The exact refresh period from DisplayConfig (read only).
    double period_ms = dm.dmDisplayFrequency ? 1000.0 / dm.dmDisplayFrequency : 1000.0 / 60.0;
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
                    std::printf("display: target \"%ls\", output technology %d, EDID ids valid %u, manufacturer 0x%04x "
                                "product 0x%04x, friendly name from EDID %u\n",
                                tn.monitorFriendlyDeviceName, static_cast<int>(tn.outputTechnology),
                                tn.flags.edidIdsValid, tn.edidManufactureId, tn.edidProductCodeId,
                                tn.flags.friendlyNameFromEdid);
                const DISPLAYCONFIG_RATIONAL &rr = paths[i].targetInfo.refreshRate;
                if (rr.Numerator && rr.Denominator) {
                    period_ms = 1000.0 * rr.Denominator / rr.Numerator;
                    std::printf("display: refresh %u/%u Hz = %.4f Hz (QueryDisplayConfig), period %.4f ms\n",
                                rr.Numerator, rr.Denominator, static_cast<double>(rr.Numerator) / rr.Denominator,
                                period_ms);
                }
            }
        }
    }
    ComPtr<IDXGIFactory5> f5;
    BOOL tearing = FALSE;
    if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f5))))
        f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing));
    std::printf("dxgi: DXGI_FEATURE_PRESENT_ALLOW_TEARING = %d\n", tearing);
    wchar_t buf[512] = {};
    DWORD sz = sizeof(buf);
    LSTATUS ls = RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\DirectX\\UserGpuPreferences",
                              L"DirectXUserGlobalSettings", RRF_RT_REG_SZ, nullptr, buf, &sz);
    std::printf("registry (read): HKCU UserGpuPreferences DirectXUserGlobalSettings = %ls\n",
                ls == ERROR_SUCCESS ? buf : L"(not set)");
    std::printf("vrr: no public API reports whether variable refresh is active on this output; see the two lines "
                "above for what can be read\n");
    // The console display state: Windows sends the current value at once after registration (read only).
    HPOWERNOTIFY pn = RegisterPowerSettingNotification(hwnd, &GUID_CONSOLE_DISPLAY_STATE, DEVICE_NOTIFY_WINDOW_HANDLE);
    for (int k = 0; k < 20 && g_display_state < 0; ++k) {
        pump();
        Sleep(10);
    }
    if (pn) UnregisterPowerSettingNotification(pn);
    std::printf("display state (GUID_CONSOLE_DISPLAY_STATE): %d (0 off, 1 on, 2 dimmed, -1 not reported)\n",
                g_display_state);
    // DWM's own frame rate over one second (DCompositionGetFrameId / DCompositionGetStatistics, read only).
    COMPOSITION_FRAME_ID f0 = 0, f1 = 0;
    LARGE_INTEGER c0, c1;
    QueryPerformanceCounter(&c0);
    const HRESULT h0 = DCompositionGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &f0);
    Sleep(1000);
    const HRESULT h1 = DCompositionGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &f1);
    QueryPerformanceCounter(&c1);
    COMPOSITION_FRAME_STATS fst = {};
    COMPOSITION_TARGET_ID tids[4] = {};
    UINT ntid = 0;
    const HRESULT h2 = DCompositionGetStatistics(f1, &fst, 4, tids, &ntid);
    std::printf("dwm: completed composition frames %llu -> %llu over %.1f ms (hr 0x%08lx 0x%08lx); last frame stats hr "
                "0x%08lx startTime %llu targetTime %llu framePeriod %llu, targets %u\n",
                static_cast<unsigned long long>(f0), static_cast<unsigned long long>(f1),
                qpc_to_ms(static_cast<UINT64>(c1.QuadPart)) - qpc_to_ms(static_cast<UINT64>(c0.QuadPart)), h0, h1, h2,
                static_cast<unsigned long long>(fst.startTime), static_cast<unsigned long long>(fst.targetTime),
                static_cast<unsigned long long>(fst.framePeriod), ntid);
    return period_ms;
}

// -------------------------------------------------------------------------------------------------------------------
// route A: composition swapchain
int route_a(const Options &o, std::vector<Sample> &samples, double &period_ms) {
    RECT client;
    HWND hwnd = make_window(o.borderless, true, &client);
    period_ms = print_context(hwnd, o);
    const UINT w = client.right - client.left, h = client.bottom - client.top;

    ComPtr<IDXGIAdapter1> adapter = pick_adapter();
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    HRESULT hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                                   nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx);
    if (FAILED(hr)) { std::printf("route A: D3D11CreateDevice failed 0x%08lx\n", hr); return 3; }

    ComPtr<IPresentationFactory> factory;
    hr = CreatePresentationFactory(dev.Get(), IID_PPV_ARGS(&factory));
    if (FAILED(hr)) { std::printf("route A: CreatePresentationFactory failed 0x%08lx\n", hr); return 3; }
    const bool supported = factory->IsPresentationSupported() != 0;
    const bool iflip_supported = factory->IsPresentationSupportedWithIndependentFlip() != 0;
    std::printf("route A: IsPresentationSupported %d, IsPresentationSupportedWithIndependentFlip %d\n", supported,
                iflip_supported);
    if (!supported) return 4;

    BOOL displayable_ok = FALSE;
    {
        ComPtr<ID3D11Device5> d5;
        D3D11_FEATURE_DATA_DISPLAYABLE fd = {};
        if (SUCCEEDED(dev.As(&d5)) && SUCCEEDED(d5->CheckFeatureSupport(D3D11_FEATURE_DISPLAYABLE, &fd, sizeof(fd))))
            displayable_ok = fd.DisplayableTexture;
    }
    std::printf("route A: D3D11_FEATURE_DISPLAYABLE DisplayableTexture %d\n", displayable_ok);

    ComPtr<IPresentationManager> pm;
    hr = factory->CreatePresentationManager(&pm);
    if (FAILED(hr)) { std::printf("route A: CreatePresentationManager failed 0x%08lx\n", hr); return 3; }

    const int kBuffers = 4;
    std::vector<ComPtr<ID3D11Texture2D>> tex(kBuffers);
    std::vector<ComPtr<ID3D11RenderTargetView>> rtv(kBuffers);
    std::vector<ComPtr<IPresentationBuffer>> pbuf(kBuffers);
    std::vector<HANDLE> avail(kBuffers);
    for (int i = 0; i < kBuffers; ++i) {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = w;
        td.Height = h;
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                       (displayable_ok ? D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE : 0);
        hr = dev->CreateTexture2D(&td, nullptr, &tex[i]);
        if (FAILED(hr)) { std::printf("route A: CreateTexture2D failed 0x%08lx\n", hr); return 3; }
        dev->CreateRenderTargetView(tex[i].Get(), nullptr, &rtv[i]);
        hr = pm->AddBufferFromResource(tex[i].Get(), &pbuf[i]);
        if (FAILED(hr)) { std::printf("route A: AddBufferFromResource failed 0x%08lx\n", hr); return 3; }
        pbuf[i]->GetAvailableEvent(&avail[i]);
    }

    HANDLE surface_handle = nullptr;
    hr = DCompositionCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, nullptr, &surface_handle);
    if (FAILED(hr)) { std::printf("route A: DCompositionCreateSurfaceHandle failed 0x%08lx\n", hr); return 3; }
    ComPtr<IPresentationSurface> surf;
    hr = pm->CreatePresentationSurface(surface_handle, &surf);
    if (FAILED(hr)) { std::printf("route A: CreatePresentationSurface failed 0x%08lx\n", hr); return 3; }
    surf->SetAlphaMode(DXGI_ALPHA_MODE_IGNORE);
    surf->SetColorSpace(DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);

    ComPtr<IDCompositionDesktopDevice> dcomp;
    hr = DCompositionCreateDevice3(nullptr, IID_PPV_ARGS(&dcomp));
    if (FAILED(hr)) { std::printf("route A: DCompositionCreateDevice3 failed 0x%08lx\n", hr); return 3; }
    ComPtr<IDCompositionTarget> target;
    dcomp->CreateTargetForHwnd(hwnd, TRUE, &target);
    ComPtr<IDCompositionVisual2> visual;
    dcomp->CreateVisual(&visual);
    ComPtr<IUnknown> dsurf;
    hr = dcomp->CreateSurfaceFromHandle(surface_handle, &dsurf);
    if (FAILED(hr)) { std::printf("route A: CreateSurfaceFromHandle failed 0x%08lx\n", hr); return 3; }
    visual->SetContent(dsurf.Get());
    target->SetRoot(visual.Get());
    dcomp->Commit();

    if (o.force_vsync) {
        const HRESULT fh = pm->ForceVSyncInterrupt(true);
        std::printf("route A: ForceVSyncInterrupt(true) hr 0x%08lx\n", fh);
    }
    pm->EnablePresentStatisticsKind(PresentStatisticsKind_PresentStatus, true);
    pm->EnablePresentStatisticsKind(PresentStatisticsKind_CompositionFrame, true);
    pm->EnablePresentStatisticsKind(PresentStatisticsKind_IndependentFlipFrame, true);
    HANDLE stats_event = nullptr;
    pm->GetPresentStatisticsAvailableEvent(&stats_event);

    const double interval_ms = period_ms * o.interval_refreshes;
    const int frames = static_cast<int>(o.seconds * 1000.0 / interval_ms);
    ULONGLONG start_it;
    QueryInterruptTimePrecise(&start_it);
    LARGE_INTEGER qf, qn;
    QueryPerformanceFrequency(&qf);
    QueryPerformanceCounter(&qn);
    std::printf("route A: clocks now: QueryInterruptTimePrecise %llu, QueryPerformanceCounter %lld at %lld Hz; "
                "target time base %s\n",
                start_it, qn.QuadPart, qf.QuadPart, o.qpc_targets ? "QPC" : "interrupt time");
    // Target times in 100 ns units on the chosen time base; it_to_ms / qpc_to_ms map them back to the run clock.
    const double ticks_per_ms = o.qpc_targets ? static_cast<double>(qf.QuadPart) / 1000.0 : 10000.0;
    const ULONGLONG base_now = o.qpc_targets ? static_cast<ULONGLONG>(qn.QuadPart) : start_it;
    const ULONGLONG first_target = base_now + static_cast<ULONGLONG>(200.0 * ticks_per_ms);  // 200 ms from now
    auto target_to_ms = [&](ULONGLONG t) { return o.qpc_targets ? qpc_to_ms(t) : it_to_ms(t); };
    auto now_ticks = [&]() -> ULONGLONG {
        if (o.qpc_targets) {
            LARGE_INTEGER c;
            QueryPerformanceCounter(&c);
            return static_cast<ULONGLONG>(c.QuadPart);
        }
        ULONGLONG it;
        QueryInterruptTimePrecise(&it);
        return it;
    };
    int n_debug = 0;
    std::map<UINT64, size_t> index;
    int n_iflip_stats = 0, n_comp_stats = 0, n_status_stats = 0;
    auto drain = [&]() {
        for (;;) {
            ComPtr<IPresentStatistics> st;
            if (pm->GetNextPresentStatistics(&st) != S_OK || !st) break;
            const UINT64 id = st->GetPresentId();
            auto it = index.find(id);
            if (it == index.end()) continue;
            Sample &s = samples[it->second];
            switch (st->GetKind()) {
            case PresentStatisticsKind_IndependentFlipFrame: {
                ComPtr<IIndependentFlipFramePresentStatistics> fs;
                if (SUCCEEDED(st.As(&fs))) {
                    s.shown_ms = it_to_ms(fs->GetDisplayedTime().value);
                    s.how = "iflip";
                    ++n_iflip_stats;
                }
                break;
            }
            case PresentStatisticsKind_CompositionFrame: {
                ComPtr<ICompositionFramePresentStatistics> cs;
                if (SUCCEEDED(st.As(&cs))) {
                    ++n_comp_stats;
                    COMPOSITION_FRAME_ID fid = cs->GetCompositionFrameId();
                    COMPOSITION_FRAME_STATS fst = {};
                    COMPOSITION_TARGET_ID tids[4] = {};
                    UINT ntid = 0;
                    if (SUCCEEDED(DCompositionGetStatistics(fid, &fst, 4, tids, &ntid)) && ntid > 0) {
                        COMPOSITION_TARGET_STATS ts = {};
                        if (SUCCEEDED(DCompositionGetTargetStatistics(fid, &tids[0], &ts)) && ts.presentTime &&
                            s.how != "iflip") {
                            s.shown_ms = qpc_to_ms(ts.presentTime);
                            s.how = "composed";
                        }
                    }
                }
                break;
            }
            case PresentStatisticsKind_PresentStatus: {
                ComPtr<IPresentStatusPresentStatistics> ps;
                if (SUCCEEDED(st.As(&ps))) {
                    ++n_status_stats;
                    const PresentStatus v = ps->GetPresentStatus();
                    if (o.verbose && n_debug < 12) {
                        ++n_debug;
                        std::printf("route A: debug present %llu status %d composition frame %llu target %.3f ms "
                                    "now %.3f ms\n",
                                    static_cast<unsigned long long>(id), static_cast<int>(v),
                                    static_cast<unsigned long long>(ps->GetCompositionFrameId()), s.target_ms,
                                    target_to_ms(now_ticks()));
                    }
                    if (v == PresentStatus_Skipped) s.how = "skipped";
                    if (v == PresentStatus_Canceled) s.how = "canceled";
                }
                break;
            }
            default: break;
            }
        }
    };

    const float grey[2][4] = {{0.20f, 0.20f, 0.22f, 1.0f}, {0.24f, 0.24f, 0.26f, 1.0f}};
    const int kAhead = 3;  // presents queued ahead of the display
    for (int i = 0; i < frames && !g_quit; ++i) {
        pump();
        drain();
        const int b = i % kBuffers;
        const ULONGLONG w0 = now_ticks();
        WaitForSingleObject(avail[b], 1000);
        if (o.verbose && i < 12)
            std::printf("route A: debug frame %d buffer %d waited %.3f ms for it, now %.3f ms\n", i, b,
                        (now_ticks() - w0) / ticks_per_ms, target_to_ms(now_ticks()));
        ctx->ClearRenderTargetView(rtv[b].Get(), grey[i & 1]);
        ctx->Flush();
        surf->SetBuffer(pbuf[b].Get());
        const ULONGLONG t = first_target + static_cast<ULONGLONG>(i * interval_ms * ticks_per_ms);
        pm->SetTargetTime(SystemInterruptTime{t});
        Sample s;
        s.id = pm->GetNextPresentId();
        s.kind = i & 1;
        s.target_ms = target_to_ms(t);
        index[s.id] = samples.size();
        samples.push_back(s);
        hr = pm->Present();
        if (FAILED(hr)) { std::printf("route A: Present failed 0x%08lx at frame %d\n", hr, i); break; }
        // Throttle: stay at most kAhead presents ahead of the target clock.
        for (;;) {
            const ULONGLONG now = now_ticks();
            if (t <= now + static_cast<ULONGLONG>(kAhead * interval_ms * ticks_per_ms)) break;
            WaitForSingleObject(stats_event, 2);
            drain();
            pump();
        }
    }
    const DWORD end = GetTickCount() + 1500;  // let the last statistics arrive
    while (GetTickCount() < end) {
        WaitForSingleObject(stats_event, 20);
        drain();
        pump();
    }
    std::printf("route A: statistics received: independent-flip %d, composition-frame %d, present-status %d\n",
                n_iflip_stats, n_comp_stats, n_status_stats);
    DestroyWindow(hwnd);
    return 0;
}

// -------------------------------------------------------------------------------------------------------------------
// route B: DXGI flip model on D3D12
int route_b(const Options &o, std::vector<Sample> &samples, double &period_ms) {
    RECT client;
    HWND hwnd = make_window(o.borderless, false, &client);
    period_ms = print_context(hwnd, o);
    const UINT w = client.right - client.left, h = client.bottom - client.top;

    ComPtr<IDXGIAdapter1> adapter = pick_adapter();
    ComPtr<ID3D12Device> dev;
    HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
    if (FAILED(hr)) { std::printf("route B: D3D12CreateDevice failed 0x%08lx\n", hr); return 3; }
    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));

    ComPtr<IDXGIFactory4> f;
    CreateDXGIFactory2(0, IID_PPV_ARGS(&f));
    const UINT kBuffers = 3;
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = w;
    sd.Height = h;
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = kBuffers;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
    ComPtr<IDXGISwapChain1> sc1;
    hr = f->CreateSwapChainForHwnd(queue.Get(), hwnd, &sd, nullptr, nullptr, &sc1);
    if (FAILED(hr)) { std::printf("route B: CreateSwapChainForHwnd failed 0x%08lx\n", hr); return 3; }
    f->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    ComPtr<IDXGISwapChain4> sc;
    sc1.As(&sc);
    sc->SetMaximumFrameLatency(2);
    HANDLE waitable = sc->GetFrameLatencyWaitableObject();

    D3D12_DESCRIPTOR_HEAP_DESC hd = {};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hd.NumDescriptors = kBuffers;
    ComPtr<ID3D12DescriptorHeap> heap;
    dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    std::vector<ComPtr<ID3D12Resource>> bufs(kBuffers);
    for (UINT i = 0; i < kBuffers; ++i) {
        sc->GetBuffer(i, IID_PPV_ARGS(&bufs[i]));
        D3D12_CPU_DESCRIPTOR_HANDLE hnd = heap->GetCPUDescriptorHandleForHeapStart();
        hnd.ptr += static_cast<SIZE_T>(i) * inc;
        dev->CreateRenderTargetView(bufs[i].Get(), nullptr, hnd);
    }
    std::vector<ComPtr<ID3D12CommandAllocator>> alloc(kBuffers);
    for (auto &a : alloc) dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a));
    ComPtr<ID3D12GraphicsCommandList> cl;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[0].Get(), nullptr, IID_PPV_ARGS(&cl));
    cl->Close();
    ComPtr<ID3D12Fence> fence;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    HANDLE fev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    std::vector<UINT64> fval(kBuffers, 0);
    UINT64 next_fence = 1;

    ComPtr<IDXGISwapChainMedia> media;
    const bool have_media = SUCCEEDED(sc.As(&media));
    std::map<std::string, int> mode_count;

    // GetFrameStatistics returns, for the latest present that reached the screen, its PresentCount, the refresh it
    // was shown at (PresentRefreshCount) and the latest VSync's refresh count and QPC time. A present's on-screen time
    // is SyncQPCTime - (SyncRefreshCount - PresentRefreshCount) periods.
    const double interval_refreshes = o.interval_refreshes;
    const int frames = static_cast<int>(o.seconds * 1000.0 / (period_ms * interval_refreshes));
    const float grey[2][4] = {{0.20f, 0.20f, 0.22f, 1.0f}, {0.24f, 0.24f, 0.26f, 1.0f}};
    std::map<UINT, size_t> by_present_count;
    int stats_fail = 0;
    auto read_stats = [&]() {
        DXGI_FRAME_STATISTICS fs = {};
        const HRESULT shr = sc->GetFrameStatistics(&fs);
        static int n_debug_b = 0;
        if (o.verbose && n_debug_b < 15) {
            ++n_debug_b;
            UINT last = 0;
            sc->GetLastPresentCount(&last);
            std::printf("route B: debug GetFrameStatistics hr 0x%08lx PresentCount %u PresentRefreshCount %u "
                        "SyncRefreshCount %u SyncQPCTime %lld; GetLastPresentCount %u\n",
                        shr, fs.PresentCount, fs.PresentRefreshCount, fs.SyncRefreshCount, fs.SyncQPCTime.QuadPart,
                        last);
        }
        if (FAILED(shr)) { ++stats_fail; return; }
        auto it = by_present_count.find(fs.PresentCount);
        if (it == by_present_count.end()) return;
        Sample &s = samples[it->second];
        if (s.shown_ms >= 0) return;
        const double sync_ms = qpc_to_ms(static_cast<UINT64>(fs.SyncQPCTime.QuadPart));
        s.shown_ms = sync_ms - static_cast<double>(fs.SyncRefreshCount - fs.PresentRefreshCount) * period_ms;
        s.how = "dxgi";
        if (have_media) {
            DXGI_FRAME_STATISTICS_MEDIA fm = {};
            if (SUCCEEDED(media->GetFrameStatisticsMedia(&fm))) {
                const char *m = fm.CompositionMode == DXGI_FRAME_PRESENTATION_MODE_COMPOSED      ? "composed"
                                : fm.CompositionMode == DXGI_FRAME_PRESENTATION_MODE_OVERLAY      ? "overlay"
                                : fm.CompositionMode == DXGI_FRAME_PRESENTATION_MODE_NONE         ? "none"
                                : fm.CompositionMode == DXGI_FRAME_PRESENTATION_MODE_COMPOSITION_FAILURE ? "failure"
                                                                                                    : "?";
                mode_count[m]++;
            }
        }
    };

    for (int i = 0; i < frames && !g_quit; ++i) {
        pump();
        WaitForSingleObjectEx(waitable, 1000, TRUE);
        const UINT b = sc->GetCurrentBackBufferIndex();
        if (fence->GetCompletedValue() < fval[b]) {
            fence->SetEventOnCompletion(fval[b], fev);
            WaitForSingleObject(fev, 1000);
        }
        alloc[b]->Reset();
        cl->Reset(alloc[b].Get(), nullptr);
        D3D12_RESOURCE_BARRIER br = {};
        br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        br.Transition.pResource = bufs[b].Get();
        br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        br.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        br.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        cl->ResourceBarrier(1, &br);
        D3D12_CPU_DESCRIPTOR_HANDLE hnd = heap->GetCPUDescriptorHandleForHeapStart();
        hnd.ptr += static_cast<SIZE_T>(b) * inc;
        cl->ClearRenderTargetView(hnd, grey[i & 1], 0, nullptr);
        std::swap(br.Transition.StateBefore, br.Transition.StateAfter);
        cl->ResourceBarrier(1, &br);
        cl->Close();
        ID3D12CommandList *lists[] = {cl.Get()};
        queue->ExecuteCommandLists(1, lists);
        LARGE_INTEGER now;
        QueryPerformanceCounter(&now);
        hr = sc->Present(o.interval_refreshes, 0);
        if (FAILED(hr)) { std::printf("route B: Present failed 0x%08lx at frame %d\n", hr, i); break; }
        fval[b] = next_fence;
        queue->Signal(fence.Get(), next_fence++);
        UINT pc = 0;
        sc->GetLastPresentCount(&pc);
        Sample s;
        s.id = pc;
        s.kind = i & 1;
        s.target_ms = qpc_to_ms(static_cast<UINT64>(now.QuadPart));
        by_present_count[pc] = samples.size();
        samples.push_back(s);
        read_stats();
    }
    for (int k = 0; k < 10; ++k) {
        Sleep(static_cast<DWORD>(period_ms) + 1);
        read_stats();
    }
    std::printf("route B: GetFrameStatistics failures %d (DXGI_ERROR_FRAME_STATISTICS_DISJOINT is normal at start)\n",
                stats_fail);
    std::printf("route B: IDXGISwapChainMedia %s;", have_media ? "available" : "not available");
    for (auto &kv : mode_count) std::printf(" CompositionMode %s x%d;", kv.first.c_str(), kv.second);
    std::printf("\n");
    // Drain the queue before releasing.
    queue->Signal(fence.Get(), next_fence);
    fence->SetEventOnCompletion(next_fence, fev);
    WaitForSingleObject(fev, 2000);
    DestroyWindow(hwnd);
    return 0;
}

// -------------------------------------------------------------------------------------------------------------------
void summarize(const Options &o, const std::vector<Sample> &samples, double period_ms) {
    const double target = period_ms * o.interval_refreshes;
    std::vector<const Sample *> shown;
    std::map<std::string, int> how;
    for (const Sample &s : samples) {
        how[s.how.empty() ? "no statistic" : s.how]++;
        if (s.shown_ms >= 0) shown.push_back(&s);
    }
    std::sort(shown.begin(), shown.end(), [](const Sample *a, const Sample *b) { return a->id < b->id; });
    // Skip the first 10 presents (start-up).
    std::vector<double> iv;
    int misses = 0, repeats = 0;
    for (size_t i = 11; i < shown.size(); ++i) {
        if (shown[i]->id != shown[i - 1]->id + 1) continue;  // only consecutive presents
        const double d = shown[i]->shown_ms - shown[i - 1]->shown_ms;
        iv.push_back(d);
        if (std::fabs(d - target) > 0.5 * period_ms) ++misses;
        if (d < 0.5 * period_ms) ++repeats;
    }
    std::printf("summary: route %c, mode %s, target interval %.4f ms (%d refresh(es) of %.4f ms), presents %zu\n",
                o.route, o.borderless ? "borderless" : "window", target, o.interval_refreshes, period_ms,
                samples.size());
    std::printf("summary: per-present result:");
    for (auto &kv : how) std::printf(" %s %d;", kv.first.c_str(), kv.second);
    std::printf("\n");
    if (iv.empty()) {
        std::printf("summary: no consecutive on-screen times, so no interval statistics\n");
        return;
    }
    double mean = 0;
    for (double d : iv) mean += d;
    mean /= iv.size();
    double var = 0;
    for (double d : iv) var += (d - mean) * (d - mean);
    const double sd = std::sqrt(var / iv.size());
    std::vector<double> sorted = iv;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) { return sorted[std::min(sorted.size() - 1, static_cast<size_t>(p * sorted.size()))]; };
    std::printf("summary: on-screen interval over %zu consecutive pairs: mean %.4f ms, sd %.4f ms, p01 %.4f, p50 %.4f, "
                "p99 %.4f, min %.4f, max %.4f ms; misses (|interval - target| > half a refresh) %d; shown within half "
                "a refresh of the previous %d\n",
                iv.size(), mean, sd, pct(0.01), pct(0.5), pct(0.99), sorted.front(), sorted.back(), misses, repeats);
    if (!o.out.empty()) {
        FILE *f = std::fopen(o.out.c_str(), "w");
        if (f) {
            std::fprintf(f, "present_id,kind,target_or_call_ms,shown_ms,how\n");
            for (const Sample &s : samples)
                std::fprintf(f, "%llu,%s,%.4f,%.4f,%s\n", static_cast<unsigned long long>(s.id),
                             s.kind ? "standin" : "real", s.target_ms, s.shown_ms, s.how.c_str());
            std::fclose(f);
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--route") o.route = static_cast<char>(std::toupper(next()[0]));
        else if (a == "--mode") o.borderless = next() == "borderless";
        else if (a == "--seconds") o.seconds = std::atof(next().c_str());
        else if (a == "--interval-refreshes") o.interval_refreshes = std::max(1, std::atoi(next().c_str()));
        else if (a == "--out") o.out = next();
        else if (a == "--timebase") o.qpc_targets = next() == "qpc";
        else if (a == "--force-vsync-interrupt") o.force_vsync = true;
        else if (a == "--verbose") o.verbose = true;
        else {
            std::printf("usage: fg_pacing_test --route A|B [--mode window|borderless] [--seconds S] "
                        "[--interval-refreshes N] [--out file.csv] [--timebase it|qpc] [--force-vsync-interrupt] "
                        "[--verbose]\n");
            return 1;
        }
    }
    o.seconds = std::min(std::max(o.seconds, 1.0), 30.0);
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    init_clocks();
    std::vector<Sample> samples;
    double period_ms = 1000.0 / 60.0;
    int rc = o.route == 'A' ? route_a(o, samples, period_ms) : route_b(o, samples, period_ms);
    if (rc) return rc;
    summarize(o, samples, period_ms);  // period_ms: the exact refresh period print_context read
    return 0;
}
