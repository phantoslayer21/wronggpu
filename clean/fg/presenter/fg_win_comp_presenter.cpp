// SPDX-License-Identifier: GPL-3.0-only
// Windows composition presenter (presentation manager + hardware flip queue): see fg_win_comp_presenter.h.
#include "fg_win_comp_presenter.h"

#include <dwmapi.h>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace wgfg {

using Microsoft::WRL::ComPtr;

namespace {

constexpr std::uint32_t kMaxSlots = 32;
constexpr FgTicks kGridRefresh = 10000000;            // re-read the compositor's vblank phase once a second
constexpr FgTicks kOcclusionWatchdogTicks = 2500000;  // 250 ms windowed occlusion watchdog
constexpr std::size_t kMaxSubmitted = 256;
constexpr std::size_t kMaxRealOnlyInFlight = 4;       // bound real-only queue depth in IPresentationManager
constexpr std::size_t kMaxCompInFlight = 8;           // bound generating queue depth (fits 6x + overlap)

bool ScanoutFormat(DXGI_FORMAT f) {
    return f == DXGI_FORMAT_B8G8R8A8_UNORM || f == DXGI_FORMAT_R8G8B8A8_UNORM || f == DXGI_FORMAT_R10G10B10A2_UNORM ||
           f == DXGI_FORMAT_R16G16B16A16_FLOAT;
}

DXGI_COLOR_SPACE_TYPE ColorSpaceFor(DXGI_FORMAT f, bool hdr) {
    if (f == DXGI_FORMAT_R16G16B16A16_FLOAT) return DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
    if (f == DXGI_FORMAT_R10G10B10A2_UNORM && hdr) return DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
}

// Production Windows composition back end (D3D11 + DirectComposition + IPresentationManager).
class Win32CompBackend final : public IFgCompBackend {
public:
    explicit Win32CompBackend(const FgCompPresenterDesc& desc)
        : desc_(desc), device12_(desc.device), current_width_(desc.width), current_height_(desc.height) {}

    ~Win32CompBackend() override {
        DetachVisual();
        DestroyBuffers();
        // Release all DirectComposition, presentation and D3D COM objects in dependency order BEFORE closing
        // underlying kernel handles (surface_handle_, event handles, and any retired buffer availability handles).
        dcomp_surface_.Reset();
        visual_.Reset();
        target_.Reset();
        dcomp_.Reset();
        surface_.Reset();
        prime_buffer_.Reset();
        prime_texture_.Reset();
        retiring_fence_.Reset();
        manager_.Reset();
        factory_.Reset();
        fences11_.clear();
        context11_.Reset();
        device11_.Reset();
        DrainRetiredHandles();
        if (stats_event_) CloseHandle(stats_event_);
        if (lost_event_) CloseHandle(lost_event_);
        if (fence_event_) CloseHandle(fence_event_);
        if (surface_handle_) CloseHandle(surface_handle_);
    }

    FgWinResult Initialize(std::uint32_t slot_count) {
        fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event_) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation);

        const LUID luid = device12_->GetAdapterLuid();
        const FgWinResult caps = FgQueryAdapterFlipCaps(luid, &info_.adapter);
        if (desc_.require_hw_flip_queue &&
            (!caps.Ok() || !info_.adapter.queried || !info_.adapter.hw_flip_queue_enabled))
            return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NoHardwareFlipQueue, info_.adapter.status);

        ComPtr<IDXGIFactory4> factory;
        HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
        if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterNotFound, hr);
        ComPtr<IDXGIAdapter1> adapter;
        hr = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
        if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterNotFound, hr);

        // Microsoft's composition swapchain API requires disabling internal driver threading optimizations so
        // synchronization with IPresentationManager is preserved. Keep the device multithreaded because
        // Initialize / RegisterFence / DestroyBuffers run on the game thread while presents run on the worker
        // (all serialized under mutex_).
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        const UINT d3d11_flags =
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_PREVENT_INTERNAL_THREADING_OPTIMIZATIONS;
        ComPtr<ID3D11Device> dev;
        ComPtr<ID3D11DeviceContext> ctx;
        hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, d3d11_flags, levels, 2,
                               D3D11_SDK_VERSION, &dev, nullptr, &ctx);
        if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::D3D11DeviceFailed, hr);
        if (FAILED(hr = dev.As(&device11_)) || FAILED(hr = ctx.As(&context11_)))
            return FgWinFail(FgStatus::Unsupported, FgWinRefusal::D3D11DeviceFailed, hr);

        hr = FgDCompCreatePresentationFactory(device11_.Get(), IID_PPV_ARGS(&factory_));
        if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NoPresentationManager, hr);
        info_.presentation_supported = factory_->IsPresentationSupported() != 0;
        info_.independent_flip_supported = factory_->IsPresentationSupportedWithIndependentFlip() != 0;
        if (!info_.presentation_supported) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NoPresentationManager);
        if (!info_.independent_flip_supported) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NoIndependentFlip);

        D3D11_FEATURE_DATA_DISPLAYABLE disp = {};
        hr = device11_->CheckFeatureSupport(D3D11_FEATURE_DISPLAYABLE, &disp, sizeof(disp));
        info_.d3d11_displayable = SUCCEEDED(hr) && disp.DisplayableTexture;
        if (!info_.d3d11_displayable) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NotDisplayable, hr);

        hr = factory_->CreatePresentationManager(&manager_);
        if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NoPresentationManager, hr);
        (void)manager_->GetPresentRetiringFence(IID_PPV_ARGS(&retiring_fence_));

        FgWinResult r = CreateBuffers(slot_count, desc_.width, desc_.height, &info_);
        if (!r.Ok()) return r;

        hr = FgDCompCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, nullptr, &surface_handle_);
        if (FAILED(hr)) return FgWinFail(FgStatus::Refused, FgWinRefusal::CompositionTarget, hr);
        hr = manager_->CreatePresentationSurface(surface_handle_, &surface_);
        if (FAILED(hr)) return FgWinFail(FgStatus::Refused, FgWinRefusal::CompositionTarget, hr);
        surface_->SetTag(1);
        if (FAILED(hr = surface_->SetAlphaMode(DXGI_ALPHA_MODE_IGNORE)) ||
            FAILED(hr = surface_->SetColorSpace(ColorSpaceFor(desc_.format, desc_.hdr))) ||
            FAILED(hr = SetSourceRect(desc_.width, desc_.height)))
            return FgWinFail(FgStatus::Refused, FgWinRefusal::CompositionTarget, hr);
        if (desc_.hdr || desc_.format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
            ComPtr<IPresentationSurface2> surf2;
            if (SUCCEEDED(surface_.As(&surf2)) && surf2) surf2->SetIsHdrContent(1);
        }

        info_.hwnd_is_root = !desc_.bind_window || (desc_.hwnd && ::GetAncestor(desc_.hwnd, GA_ROOT) == desc_.hwnd);
        info_.window_cloaked = IsWindowCloaked();

        if (desc_.bind_window) {
            // Prime the presentation surface with a dedicated opaque displayable buffer BEFORE committing the
            // DirectComposition visual so DWM's first view of the topmost visual is opaque and full-window (not empty).
            D3D11_TEXTURE2D_DESC ptd = {};
            ptd.Width = desc_.width;
            ptd.Height = desc_.height;
            ptd.MipLevels = 1;
            ptd.ArraySize = 1;
            ptd.Format = desc_.format;
            ptd.SampleDesc.Count = 1;
            ptd.Usage = D3D11_USAGE_DEFAULT;
            ptd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            ptd.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                            D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE;
            if (SUCCEEDED(device11_->CreateTexture2D(&ptd, nullptr, &prime_texture_)) && prime_texture_) {
                ComPtr<ID3D11RenderTargetView> prtv;
                if (SUCCEEDED(device11_->CreateRenderTargetView(prime_texture_.Get(), nullptr, &prtv)) && prtv) {
                    const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
                    context11_->ClearRenderTargetView(prtv.Get(), black);
                    context11_->Flush();
                }
                if (SUCCEEDED(manager_->AddBufferFromResource(prime_texture_.Get(), &prime_buffer_)) && prime_buffer_) {
                    const UINT64 ppid = manager_->GetNextPresentId();
                    if (SUCCEEDED(surface_->SetBuffer(prime_buffer_.Get())) &&
                        SUCCEEDED(manager_->SetTargetTime(SystemInterruptTime{0})) &&
                        SUCCEEDED(manager_->Present())) {
                        prime_present_id_ = ppid;
                        info_.primed = true;
                    } else {
                        prime_buffer_.Reset();
                        prime_texture_.Reset();
                    }
                } else {
                    prime_texture_.Reset();
                }
            }

            ComPtr<IDXGIDevice> dxgi_dev;
            device11_.As(&dxgi_dev);
            if (FAILED(hr = FgDCompCreateDevice3(dxgi_dev.Get(), IID_PPV_ARGS(&dcomp_))) ||
                FAILED(hr = dcomp_->CreateTargetForHwnd(desc_.hwnd, TRUE, &target_)) ||
                FAILED(hr = dcomp_->CreateVisual(&visual_)) ||
                FAILED(hr = dcomp_->CreateSurfaceFromHandle(surface_handle_, &dcomp_surface_)) ||
                FAILED(hr = visual_->SetContent(dcomp_surface_.Get())) ||
                FAILED(hr = target_->SetRoot(visual_.Get())) ||
                FAILED(hr = dcomp_->Commit()))
                return FgWinFail(FgStatus::Refused, FgWinRefusal::CompositionTarget, hr);
            // Wait for DWM to apply the visual tree before the first session Present is issued.
            dcomp_->WaitForCommitCompletion();
        }

        if (FAILED(hr = manager_->EnablePresentStatisticsKind(PresentStatisticsKind_PresentStatus, 1)) ||
            FAILED(hr = manager_->EnablePresentStatisticsKind(PresentStatisticsKind_CompositionFrame, 1)) ||
            FAILED(hr = manager_->EnablePresentStatisticsKind(PresentStatisticsKind_IndependentFlipFrame, 1)) ||
            FAILED(hr = manager_->GetPresentStatisticsAvailableEvent(&stats_event_)) ||
            FAILED(hr = manager_->GetLostEvent(&lost_event_)))
            return FgWinFail(FgStatus::Refused, FgWinRefusal::StatisticsUnavailable, hr);

        return FgWinOk();
    }

    FgTicks Now() const override { return FgInterruptNow(); }
    HANDLE StatsEvent() const override { return stats_event_; }
    HANDLE LostEvent() const override { return lost_event_; }
    HANDLE FenceEvent() const override { return fence_event_; }

    HANDLE SlotAvailableEvent(std::uint32_t slot) const override {
        return slot < buffers_.size() ? buffers_[slot].available : nullptr;
    }

    bool IsLost() const override {
        return (lost_event_ && WaitForSingleObject(lost_event_, 0) == WAIT_OBJECT_0) ||
               (device12_ && device12_->GetDeviceRemovedReason() != S_OK);
    }

    bool IsWindowVisible() const override {
        if (!desc_.bind_window || !desc_.hwnd || !::IsWindow(desc_.hwnd) || !::IsWindowVisible(desc_.hwnd) ||
            ::IsIconic(desc_.hwnd))
            return false;
        RECT cr = {};
        return ::GetClientRect(desc_.hwnd, &cr) != FALSE && cr.right > cr.left && cr.bottom > cr.top;
    }

    bool IsWindowCloaked() const override {
        if (!desc_.bind_window || !desc_.hwnd || !::IsWindow(desc_.hwnd)) return false;
        using DwmGetAttrFn = HRESULT(WINAPI*)(HWND, DWORD, PVOID, DWORD);
        static const DwmGetAttrFn dwm_get_attr = []() -> DwmGetAttrFn {
            const HMODULE m = ::LoadLibraryExW(L"dwmapi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
            return m ? reinterpret_cast<DwmGetAttrFn>(
                           reinterpret_cast<void*>(::GetProcAddress(m, "DwmGetWindowAttribute")))
                     : nullptr;
        }();
        if (!dwm_get_attr) return false;
        DWORD cloaked = 0;
        return SUCCEEDED(dwm_get_attr(desc_.hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked))) && cloaked != 0;
    }

    std::uint64_t RetiringFenceValue() const override {
        return retiring_fence_ ? static_cast<std::uint64_t>(retiring_fence_->GetCompletedValue()) : 0u;
    }

    std::uint64_t PrimePresentId() const override { return prime_present_id_; }

    std::uint64_t CompletedCompositionFrameId() const override {
        COMPOSITION_FRAME_ID id = 0;
        return SUCCEEDED(FgDCompGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &id)) ? static_cast<std::uint64_t>(id) : 0u;
    }

    bool ProbeCompFrameTiming(std::uint64_t cfid, FgCompFrameTiming* out) const override {
        if (cfid == 0 || !out) return false;
        *out = FgCompFrameTiming{};
        COMPOSITION_FRAME_STATS fst = {};
        COMPOSITION_TARGET_ID targets[4] = {};
        UINT count = 0;
        if (FAILED(FgDCompGetStatistics(cfid, &fst, 4, targets, &count))) return false;
        out->target_count = count;
        if (count > 0) {
            COMPOSITION_TARGET_STATS ts = {};
            if (SUCCEEDED(FgDCompGetTargetStatistics(cfid, &targets[0], &ts))) {
                out->raw_completed_qpc = ts.completedStats.time;
                out->raw_presented_qpc = ts.presentedStats.time;
                if (ts.completedStats.time != 0)
                    out->completed_time = clock_map_.FromQpc(static_cast<std::int64_t>(ts.completedStats.time));
                if (ts.presentedStats.time != 0)
                    out->presented_time = clock_map_.FromQpc(static_cast<std::int64_t>(ts.presentedStats.time));
                if (ts.presentTime != 0)
                    out->target_present_time = clock_map_.FromQpc(static_cast<std::int64_t>(ts.presentTime));
            }
        }
        return true;
    }

    const IFgFenceSource* FenceSource() const override { return &fences12_; }

    void WatchFence(const FgReadyFence& fence) override {
        if (!fence_event_) return;
        if (ID3D12Fence* f = fences12_.Get(fence.fence)) f->SetEventOnCompletion(fence.value, fence_event_);
    }

    std::uint32_t RegisterFence(ID3D12Fence* fence, FgWinResult* why) override {
        if (why) *why = FgWinOk();
        if (!fence) {
            if (why) *why = FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullDevice);
            return 0;
        }
        ComPtr<ID3D12Fence1> extended;
        if (SUCCEEDED(fence->QueryInterface(IID_PPV_ARGS(&extended))) &&
            !(extended->GetCreationFlags() & D3D12_FENCE_FLAG_SHARED)) {
            if (why) *why = FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::SharedHandle, E_INVALIDARG);
            return 0;
        }
        HANDLE shared = nullptr;
        HRESULT hr = device12_->CreateSharedHandle(fence, nullptr, GENERIC_ALL, nullptr, &shared);
        if (FAILED(hr)) {  // a fence created without D3D12_FENCE_FLAG_SHARED
            if (why) *why = FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::SharedHandle, hr);
            return 0;
        }
        ComPtr<ID3D11Fence> f11;
        hr = device11_->OpenSharedFence(shared, IID_PPV_ARGS(&f11));
        CloseHandle(shared);
        if (FAILED(hr)) {
            if (why) *why = FgWinFail(FgStatus::Unsupported, FgWinRefusal::SharedHandle, hr);
            return 0;
        }
        const std::uint32_t id = fences12_.Add(fence);
        if (id == 0) {
            if (why) *why = FgWinFail(FgStatus::NoSlot, FgWinRefusal::ResourceCreation);
            return 0;
        }
        if (fences11_.size() < id) fences11_.resize(id);
        fences11_[id - 1] = f11;
        return id;
    }

    ID3D12Resource* SlotResource(std::uint32_t slot) const override {
        return slot < buffers_.size() ? buffers_[slot].resource.Get() : nullptr;
    }

    FgCompBackendInfo BackendInfo() const override { return info_; }

    FgWinResult CreateBuffers(std::uint32_t slot_count, UINT width, UINT height, FgCompBackendInfo* out_info) override {
        buffers_.resize(slot_count);
        for (std::uint32_t i = 0; i < slot_count; ++i) {
            Buffer& b = buffers_[i];
            D3D12_HEAP_PROPERTIES hp = {};
            hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            hp.CreationNodeMask = hp.VisibleNodeMask = 1;
            D3D12_RESOURCE_DESC rd = {};
            rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            rd.Width = width;
            rd.Height = height;
            rd.DepthOrArraySize = 1;
            rd.MipLevels = 1;
            rd.Format = desc_.format;
            rd.SampleDesc.Count = 1;
            rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
            HRESULT hr = device12_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_DISPLAY,
                                                            &rd, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&b.resource));
            if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation, hr);
            info_.simultaneous_access = true;
            HANDLE shared = nullptr;
            hr = device12_->CreateSharedHandle(b.resource.Get(), nullptr, GENERIC_ALL, nullptr, &shared);
            if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::SharedHandle, hr);
            hr = device11_->OpenSharedResource1(shared, IID_PPV_ARGS(&b.texture));
            CloseHandle(shared);
            if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::SharedHandle, hr);
            D3D11_TEXTURE2D_DESC td = {};
            b.texture->GetDesc(&td);
            info_.d3d11_misc_flags = td.MiscFlags;
            if (!(td.MiscFlags & D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE))
                return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NotDisplayable);
            hr = manager_->AddBufferFromResource(b.texture.Get(), &b.buffer);
            if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NotDisplayable, hr);
            hr = b.buffer->GetAvailableEvent(&b.available);
            if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation, hr);
        }
        current_width_ = width;
        current_height_ = height;
        if (out_info) *out_info = info_;
        return FgWinOk();
    }

    void DestroyBuffers() override {
        // Defer closing b.available until the worker thread exits WaitForMultipleObjects (DrainRetiredHandles)
        // so Detach() / CompleteResize() never closes a HANDLE while Run() is waiting on it.
        for (Buffer& b : buffers_) {
            if (b.available) {
                retired_handles_.push_back(b.available);
                b.available = nullptr;
            }
            b = Buffer();
        }
        buffers_.clear();
    }

    void DrainRetiredHandles() override {
        for (HANDLE h : retired_handles_) {
            if (h) CloseHandle(h);
        }
        retired_handles_.clear();
    }

    HRESULT SetSourceRect(UINT width, UINT height) override {
        current_width_ = width;
        current_height_ = height;
        if (!surface_) return E_UNEXPECTED;
        const RECT src = {0, 0, static_cast<LONG>(width), static_cast<LONG>(height)};
        HRESULT hr = surface_->SetSourceRect(&src);
        if (FAILED(hr)) return hr;
        return ApplySurfaceTransform(width, height);
    }

    void DetachVisual() override {
        if (detached_) return;
        detached_ = true;
        // Detach only the DirectComposition visual tree and commit asynchronously. Never call
        // surface_->SetBuffer(nullptr) + manager_->Present() (which access-violates after DestroyBuffers()) and
        // never block the HWND owner thread in WaitForCommitCompletion() during teardown.
        if (visual_) (void)visual_->SetContent(nullptr);
        if (target_) (void)target_->SetRoot(nullptr);
        if (dcomp_) (void)dcomp_->Commit();
    }

    bool IsBufferAvailable(std::uint32_t slot) const override {
        if (slot >= buffers_.size() || !buffers_[slot].buffer) return false;
        boolean available = 0;
        return SUCCEEDED(buffers_[slot].buffer->IsAvailable(&available)) && available != 0;
    }

    HRESULT ForceVSyncInterrupt(bool enable) override {
        return manager_ ? manager_->ForceVSyncInterrupt(enable ? 1 : 0) : E_UNEXPECTED;
    }

    HRESULT PresentSlot(std::uint32_t slot, FgTicks target_time, const FgReadyFence& fence,
                        std::uint64_t* platform_id_out) override {
        if (slot >= buffers_.size() || !buffers_[slot].buffer || !surface_ || !manager_) return E_UNEXPECTED;
        // GPU-side order: the presentation device waits for the producer fence before the present (already complete
        // on the CPU; this keeps the order explicit on the GPU timeline too).
        if (fence.fence >= 1 && fence.fence <= fences11_.size() && fences11_[fence.fence - 1])
            context11_->Wait(fences11_[fence.fence - 1].Get(), fence.value);
        RefreshClientTransformIfNeeded();
        HRESULT hr = surface_->SetBuffer(buffers_[slot].buffer.Get());
        if (SUCCEEDED(hr)) hr = manager_->SetTargetTime(SystemInterruptTime{static_cast<UINT64>(target_time)});
        const std::uint64_t pid = manager_->GetNextPresentId();
        if (SUCCEEDED(hr)) hr = manager_->Present();
        if (SUCCEEDED(hr) && platform_id_out) *platform_id_out = pid;
        return hr;
    }

    HRESULT CancelPresentsFrom(std::uint64_t platform_id) override {
        return manager_ ? manager_->CancelPresentsFrom(platform_id) : S_OK;
    }

    bool PopStatistic(FgCompStatistic* out) override {
        if (!out || !manager_) return false;
        *out = FgCompStatistic{};
        ComPtr<IPresentStatistics> st;
        if (manager_->GetNextPresentStatistics(&st) != S_OK || !st) return false;
        out->present_id = st->GetPresentId();
        switch (st->GetKind()) {
            case PresentStatisticsKind_PresentStatus: {
                ComPtr<IPresentStatusPresentStatistics> ps;
                if (FAILED(st.As(&ps))) return true;
                out->kind = FgCompStatKind::PresentStatus;
                const PresentStatus v = ps->GetPresentStatus();
                if (v == PresentStatus_Skipped) out->status = FgCompPresentStatus::Skipped;
                else if (v == PresentStatus_Canceled) out->status = FgCompPresentStatus::Canceled;
                else out->status = FgCompPresentStatus::Queued;
                const COMPOSITION_FRAME_ID cfid = ps->GetCompositionFrameId();
                out->comp_frame_id = cfid;
                if (cfid != 0 && status_probes_ < 16) {
                    COMPOSITION_FRAME_STATS fst = {};
                    COMPOSITION_TARGET_ID targets[4] = {};
                    UINT count = 0;
                    if (SUCCEEDED(FgDCompGetStatistics(cfid, &fst, 4, targets, &count))) {
                        ++status_probes_;
                        out->status_target_count = count;
                        if (count > 0) {
                            COMPOSITION_TARGET_STATS ts = {};
                            if (SUCCEEDED(FgDCompGetTargetStatistics(cfid, &targets[0], &ts))) {
                                out->status_target_present_time = ts.presentTime;
                                out->status_target_presented_time = ts.presentedStats.time;
                                out->status_target_completed_time = ts.completedStats.time;
                            }
                        }
                    }
                }
                return true;
            }
            case PresentStatisticsKind_IndependentFlipFrame: {
                ComPtr<IIndependentFlipFramePresentStatistics> fs;
                if (FAILED(st.As(&fs))) return true;
                out->kind = FgCompStatKind::IndependentFlipFrame;
                const FgTicks timestamp = static_cast<FgTicks>(fs->GetDisplayedTime().value);
                out->displayed_time = timestamp;
                out->displayed_measured = timestamp != 0;
                return true;
            }
            case PresentStatisticsKind_CompositionFrame: {
                ComPtr<ICompositionFramePresentStatistics> cs;
                out->kind = FgCompStatKind::CompositionFrame;
                if (SUCCEEDED(st.As(&cs))) {
                    const COMPOSITION_FRAME_ID fid = cs->GetCompositionFrameId();
                    out->comp_frame_id = fid;
                    FgCompFrameTiming timing;
                    if (ProbeCompFrameTiming(fid, &timing) && timing.target_count > 0) {
                        // Prefer completedStats.time, then presentedStats.time (both measured DComp target timestamps
                        // in QPC, mapped into SystemInterruptTime). Target presentTime alone is not measured.
                        if (timing.completed_time > 0) {
                            out->displayed_time = timing.completed_time;
                            out->displayed_measured = true;
                        } else if (timing.presented_time > 0) {
                            out->displayed_time = timing.presented_time;
                            out->displayed_measured = true;
                        } else if (timing.target_present_time > 0) {
                            out->displayed_time = timing.target_present_time;
                            out->displayed_measured = false;
                        }
                    }
                }
                return true;
            }
            default:
                return true;
        }
    }

    bool ReadRefreshGrid(FgTicks period_override, FgRefreshGrid* out) override {
        return FgReadCompositorGrid(clock_map_, period_override, out);
    }

private:
    struct Buffer {
        ComPtr<ID3D12Resource> resource;
        ComPtr<ID3D11Texture2D> texture;
        ComPtr<IPresentationBuffer> buffer;
        HANDLE available = nullptr;
    };

    HRESULT ApplySurfaceTransform(UINT width, UINT height) {
        if (!surface_ || width == 0 || height == 0) return S_OK;
        PresentationTransform tx = {};
        tx.M11 = 1.0f;
        tx.M22 = 1.0f;
        if (desc_.bind_window && desc_.hwnd) {
            RECT cr = {};
            if (::GetClientRect(desc_.hwnd, &cr)) {
                const LONG cw = cr.right - cr.left;
                const LONG ch = cr.bottom - cr.top;
                if (cw > 0 && ch > 0) {
                    tx.M11 = static_cast<float>(cw) / static_cast<float>(width);
                    tx.M22 = static_cast<float>(ch) / static_cast<float>(height);
                    last_client_w_ = cw;
                    last_client_h_ = ch;
                }
            }
        }
        return surface_->SetTransform(&tx);
    }

    void RefreshClientTransformIfNeeded() {
        if (!desc_.bind_window || !desc_.hwnd || !surface_ || current_width_ == 0 || current_height_ == 0) return;
        RECT cr = {};
        if (!::GetClientRect(desc_.hwnd, &cr)) return;
        const LONG cw = cr.right - cr.left;
        const LONG ch = cr.bottom - cr.top;
        if (cw > 0 && ch > 0 && (cw != last_client_w_ || ch != last_client_h_)) {
            (void)ApplySurfaceTransform(current_width_, current_height_);
        }
    }

    FgCompPresenterDesc desc_;
    FgCompBackendInfo info_;
    ComPtr<ID3D12Device> device12_;
    ComPtr<ID3D11Device5> device11_;
    ComPtr<ID3D11DeviceContext4> context11_;
    ComPtr<IPresentationFactory> factory_;
    ComPtr<IPresentationManager> manager_;
    ComPtr<IPresentationSurface> surface_;
    ComPtr<ID3D11Texture2D> prime_texture_;
    ComPtr<IPresentationBuffer> prime_buffer_;
    ComPtr<ID3D11Fence> retiring_fence_;
    ComPtr<IDCompositionDesktopDevice> dcomp_;
    ComPtr<IDCompositionTarget> target_;
    ComPtr<IDCompositionVisual2> visual_;
    ComPtr<IUnknown> dcomp_surface_;
    HANDLE surface_handle_ = nullptr;
    HANDLE stats_event_ = nullptr;
    HANDLE lost_event_ = nullptr;
    HANDLE fence_event_ = nullptr;
    std::vector<Buffer> buffers_;
    std::vector<HANDLE> retired_handles_;
    FgD3D12FenceTable fences12_;
    std::vector<ComPtr<ID3D11Fence>> fences11_;  // index = fence id - 1
    FgClockMap clock_map_;
    std::uint64_t prime_present_id_ = 0;
    std::uint32_t status_probes_ = 0;
    UINT current_width_ = 0;
    UINT current_height_ = 0;
    LONG last_client_w_ = 0;
    LONG last_client_h_ = 0;
    bool detached_ = false;
};

}  // namespace

FgWinCompPresenter::FgWinCompPresenter(const FgCompPresenterDesc& desc, const FgEngineConfig& engine,
                                       std::unique_ptr<IFgCompBackend> backend, bool start_worker)
    : desc_(desc),
      backend_(std::move(backend)),
      start_worker_(start_worker),
      engine_(engine, backend_->FenceSource()),
      display_(desc.refresh_period) {
    info_.slot_count = engine.slot_count;
}

FgWinResult FgWinCompPresenter::Create(const FgCompPresenterDesc& d, std::unique_ptr<FgWinCompPresenter>* out) {
    if (!out) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::None);
    out->reset();
    // The window first: an invalid or foreign HWND is refused before anything is created.
    if (d.bind_window) {
        if (!d.hwnd || !IsWindow(d.hwnd)) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::InvalidWindow);
        DWORD pid = 0;
        GetWindowThreadProcessId(d.hwnd, &pid);
        if (pid != GetCurrentProcessId()) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::ForeignWindow);
    } else if (d.hwnd) {
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::InvalidWindow);  // headless mode binds nothing
    }
    // plain arguments before the device, so every argument refusal is decided without touching a driver
    if (d.max_generated < 1 || d.max_generated > kFgMaxGenerated)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::BadCount);
    const std::uint32_t slots = d.slot_count ? d.slot_count : d.max_generated + 4;
    if (slots < FgMinSlotsFor(d.max_generated) || slots > kMaxSlots)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::SlotCount);
    if (d.width < 1 || d.height < 1 || d.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        d.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::BadSize);
    if (!ScanoutFormat(d.format)) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::BadFormat);
    if (d.refresh_period < 0) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::RefreshUnknown);
    if (!d.device) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullDevice);

    std::unique_ptr<Win32CompBackend> backend(new Win32CompBackend(d));
    const FgWinResult init = backend->Initialize(slots);
    if (!init.Ok()) return init;
    return CreateWithBackend(d, std::move(backend), true, out);
}

FgWinResult FgWinCompPresenter::CreateWithBackend(const FgCompPresenterDesc& d,
                                                  std::unique_ptr<IFgCompBackend> backend, bool start_worker,
                                                  std::unique_ptr<FgWinCompPresenter>* out) {
    if (!out) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::None);
    out->reset();
    if (!backend) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullDevice);
    if (d.max_generated < 1 || d.max_generated > kFgMaxGenerated)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::BadCount);
    const std::uint32_t slots = d.slot_count ? d.slot_count : d.max_generated + 4;
    if (slots < FgMinSlotsFor(d.max_generated) || slots > kMaxSlots)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::SlotCount);
    if (d.width < 1 || d.height < 1 || d.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        d.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::BadSize);
    if (!ScanoutFormat(d.format)) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::BadFormat);
    if (d.refresh_period < 0) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::RefreshUnknown);

    FgEngineConfig cfg;
    cfg.probe = FgPresenterProbe{FgPresenterBackend::WindowsComposition, true, true, d.max_generated, FgStatus::Ok};
    cfg.slot_count = slots;
    cfg.max_pending_intervals = d.max_pending_intervals >= 1 ? d.max_pending_intervals : 2u;
    cfg.refresh_period = d.refresh_period;
    if (d.variable_refresh) cfg.pending_late_tolerance = 40000;  // 4 ms envelope matching FgPlanTearingImage
    std::unique_ptr<FgWinCompPresenter> p(new FgWinCompPresenter(d, cfg, std::move(backend), start_worker));
    const FgWinResult r = p->Initialize();
    if (!r.Ok()) return r;
    *out = std::move(p);
    return FgWinOk();
}

FgWinResult FgWinCompPresenter::Initialize() {
    if (start_worker_ && !timer_.Valid()) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::TimerUnavailable);
    const FgCompBackendInfo bi = backend_->BackendInfo();
    info_.adapter = bi.adapter;
    info_.presentation_supported = bi.presentation_supported;
    info_.independent_flip_supported = bi.independent_flip_supported;
    info_.d3d11_displayable = bi.d3d11_displayable;
    info_.d3d11_misc_flags = bi.d3d11_misc_flags;
    info_.simultaneous_access = bi.simultaneous_access;
    info_.primed = bi.primed;
    info_.hwnd_is_root = bi.hwnd_is_root;
    info_.window_visible = backend_->IsWindowVisible();
    info_.window_cloaked = backend_->IsWindowCloaked();
    info_.retiring_fence_value = backend_->RetiringFenceValue();

    const HRESULT hr = backend_->ForceVSyncInterrupt(desc_.force_vsync_interrupt);
    if (FAILED(hr)) return FgWinFail(FgStatus::Refused, FgWinRefusal::StatisticsUnavailable, hr);
    info_.force_vsync_interrupt = desc_.force_vsync_interrupt;

    if (!desc_.variable_refresh) {
        UpdateGridLocked();  // no worker yet: nothing to lock against
        if (info_.grid.period <= 0) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::RefreshUnknown);
    }
    if (start_worker_ && !worker_.Start(this, L"wg-fg-composition-presenter", desc_.worker_registration))
        return FgWinFail(FgStatus::Refused, FgWinRefusal::WorkerStart);
    return FgWinOk();
}

FgWinCompPresenter::~FgWinCompPresenter() {
    worker_.Stop();
    std::lock_guard<std::mutex> lock(mutex_);
    if (claimed_) FgChainRegistry::Process().Release(chain_, this);
    claimed_ = false;
    if (backend_) {
        if (!submitted_.empty()) (void)backend_->CancelPresentsFrom(submitted_.front().platform_id);
        backend_->DetachVisual();
        backend_->DestroyBuffers();
        backend_->DrainRetiredHandles();
    }
}

void FgWinCompPresenter::StepForTest() {
    std::lock_guard<std::mutex> lock(mutex_);
    StepLocked();
}

FgTicks FgWinCompPresenter::NowLocked() const { return backend_ ? backend_->Now() : FgInterruptNow(); }

FgTicks FgWinCompPresenter::RealOnlyLookahead() const {
    const FgTicks base = desc_.variable_refresh
                             ? (desc_.min_interval > 0 ? desc_.min_interval
                                                       : (desc_.refresh_period > 0 ? desc_.refresh_period : 10000))
                             : (info_.grid.period > 0 ? info_.grid.period : 166667);
    return std::min<FgTicks>(std::max<FgTicks>(base / 2, 5000), 20000);
}

FgTicks FgWinCompPresenter::PresentLookaheadLocked(bool real_only_plan) const {
    if (real_only_plan) return RealOnlyLookahead();
    // Once hardware Independent Flip is active, allow deeper future target submission into the hardware flip queue.
    // In composed mode DWM does not hold a multi-frame future flip queue per visual (submitting a whole interval at
    // once causes later images in the burst to skip earlier ones and display early), so pace every image to ~half a
    // refresh ahead of its target time.
    if (info_.indep_flip_frames >= 4) return 0;  // 0 = unbounded lookahead (up to kMaxCompInFlight)
    const FgTicks base = desc_.variable_refresh
                             ? (desc_.min_interval > 0 ? desc_.min_interval
                                                       : (desc_.refresh_period > 0 ? desc_.refresh_period : 10000))
                             : (info_.grid.period > 0 ? info_.grid.period : 166667);
    return std::min<FgTicks>(std::max<FgTicks>(base / 2, 5000), 15000);
}

FgTicks FgWinCompPresenter::RefreshPeriodLocked() const {
    if (desc_.refresh_period > 0) return desc_.refresh_period;
    if (info_.grid.period > 0) return info_.grid.period;
    return 166667;
}

void FgWinCompPresenter::SetRealRetireCallback(FgRealRetireFn fn, void* ctx) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.SetRealRetireCallback(fn, ctx);
}

void FgWinCompPresenter::SetTraceCallback(FgPresentTraceFn fn, void* ctx) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.SetTraceCallback(fn, ctx);
}

std::uint32_t FgWinCompPresenter::RegisterFence(ID3D12Fence* fence, FgWinResult* why) {
    std::lock_guard<std::mutex> lock(mutex_);
    return backend_ ? backend_->RegisterFence(fence, why) : 0u;
}

ID3D12Resource* FgWinCompPresenter::SlotResource(std::uint32_t slot) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backend_ ? backend_->SlotResource(slot) : nullptr;
}

FgStatus FgWinCompPresenter::ReserveRealSlot(std::uint32_t* slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.ReserveRealOnlySlot(slot);
}

void FgWinCompPresenter::ReleaseRealSlot() {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.ReleaseRealOnlySlot();
}

FgStatus FgWinCompPresenter::SetResizeTarget(std::uint32_t width, std::uint32_t height) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (width < 1 || height < 1 || width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
        height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION)
        return engine_.Refuse(FgStatus::InvalidArgument);
    resize_width_ = width;
    resize_height_ = height;
    return FgStatus::Ok;
}

FgPresenterProbe FgWinCompPresenter::Probe() const {
    return FgPresenterProbe{FgPresenterBackend::WindowsComposition, true, true, desc_.max_generated, FgStatus::Ok};
}

FgCompPresenterInfo FgWinCompPresenter::Info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    FgCompPresenterInfo out = info_;
    if (backend_) {
        out.window_cloaked = backend_->IsWindowCloaked();
        out.retiring_fence_value = backend_->RetiringFenceValue();
        if (out.status_first_cfid != 0 &&
            (out.status_target_count_max == 0 ||
             (out.status_target_completed_nonzero == 0 && out.status_target_presented_nonzero == 0))) {
            FgCompFrameTiming timing;
            if (backend_->ProbeCompFrameTiming(out.status_first_cfid, &timing)) {
                if (timing.target_count > out.status_target_count_max)
                    out.status_target_count_max = timing.target_count;
                if (timing.raw_presented_qpc != 0 && out.status_target_presented_nonzero == 0)
                    out.status_target_presented_nonzero = 1;
                if (timing.raw_completed_qpc != 0 && out.status_target_completed_nonzero == 0)
                    out.status_target_completed_nonzero = 1;
            }
        }
    }
    return out;
}

FgDisplaySummary FgWinCompPresenter::ReadDisplayCadence(bool window) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = window ? display_.TakeWindow() : display_.Session();
    result.pending = submitted_.size();
    return result;
}

std::uint32_t FgWinCompPresenter::FreeSlots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.FreeSlots();
}

FgPresenterState FgWinCompPresenter::State() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.State();
}

FgPresentStats FgWinCompPresenter::Stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.Stats();
}

std::uint64_t FgWinCompPresenter::FirstQueuedPresentId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.FirstQueuedPresentId();
}

FgStatus FgWinCompPresenter::Attach(std::uint64_t chain_id, std::uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (claimed_) return engine_.Refuse(FgStatus::WrongState);
    if (engine_.State() != FgPresenterState::Detached) return engine_.Refuse(FgStatus::WrongState);
    if (need_buffers_) {  // the previous detach handed its buffers to the compositor
        backend_->DestroyBuffers();
        FgCompBackendInfo bi;
        if (!backend_->CreateBuffers(info_.slot_count, desc_.width, desc_.height, &bi).Ok())
            return engine_.Refuse(FgStatus::Refused);
        info_.d3d11_misc_flags = bi.d3d11_misc_flags;
        info_.simultaneous_access = bi.simultaneous_access;
        need_buffers_ = false;
        worker_.Wake();
    }
    if (!FgChainRegistry::Process().Claim(chain_id, this)) return engine_.Refuse(FgStatus::Refused);
    const FgStatus s = engine_.Attach(epoch);
    if (s != FgStatus::Ok) {
        FgChainRegistry::Process().Release(chain_id, this);
        return s;
    }
    chain_ = chain_id;
    claimed_ = true;
    have_vblank_ = false;
    next_pace_wake_ = 0;
    return s;
}

FgStatus FgWinCompPresenter::BeginInterval(const FgIntervalPlan& plan, FgIntervalReservation* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    // fixed refresh: an image interval that is not a whole number of refreshes cannot be shown evenly; refuse it
    // (the scheduler presents real-only or changes the count) instead of dropping images on the display path
    if (!desc_.variable_refresh && engine_.State() == FgPresenterState::Attached &&
        !FgPlanFitsRefresh(plan, info_.grid.period)) {
        if (out) *out = FgIntervalReservation();
        return engine_.Refuse(FgStatus::Unsupported);
    }
    return engine_.BeginInterval(plan, NowLocked(), out);
}

FgStatus FgWinCompPresenter::SubmitInterval(const FgIntervalSubmission& submission) {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        s = engine_.SubmitInterval(submission, NowLocked());
    }
    worker_.Wake();
    return s;
}

FgStatus FgWinCompPresenter::PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real) {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // the image must already be in a presenter slot (ReserveRealSlot): there is no copy on this path
        if (!engine_.HasRealOnlyReservation()) return engine_.Refuse(FgStatus::WrongState);
        s = engine_.PresentRealOnly(plan, real);
    }
    worker_.Wake();
    return s;
}

FgStatus FgWinCompPresenter::RevokeGenerated(std::uint64_t interval_id, FgDropReason reason) {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::uint64_t from = 0;
        s = engine_.RevokeGenerated(interval_id, reason, NowLocked(), &from);
        CancelOnDisplayLocked(from);
    }
    worker_.Wake();
    return s;
}

std::uint32_t FgWinCompPresenter::CancelPresentsFrom(std::uint64_t present_id) {
    std::uint32_t n;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::uint64_t from = 0;
        n = engine_.CancelPresentsFrom(present_id, NowLocked(), &from);
        CancelOnDisplayLocked(from);
    }
    worker_.Wake();
    return n;
}

FgStatus FgWinCompPresenter::Quiesce() {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::uint64_t from = 0;
        s = engine_.Quiesce(NowLocked(), &from);
        CancelOnDisplayLocked(from);
    }
    worker_.Wake();
    return s;
}

FgStatus FgWinCompPresenter::CompleteResize(bool resize_ok, std::uint64_t new_epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (engine_.State() == FgPresenterState::Quiescing && resize_ok && new_epoch <= engine_.Epoch())
        return engine_.Refuse(FgStatus::InvalidArgument);
    // A resize replaces every buffer. Buffers only the display path still references (on screen, or cancelled and
    // not yet given back) are handed to the compositor and never reused; producer work still in flight is waited
    // for (Busy), since the executor may still be writing a slot.
    const bool replace = engine_.State() == FgPresenterState::Quiescing && engine_.OnlyDisplayHeld();
    if (replace) {
        backend_->DestroyBuffers();
        engine_.ForgetDisplayHeld();
        need_buffers_ = true;
    }
    const FgStatus s = engine_.CompleteResize(resize_ok, new_epoch);
    if (s != FgStatus::Ok) return s;
    const UINT w = resize_width_ ? resize_width_ : desc_.width;
    const UINT h = resize_height_ ? resize_height_ : desc_.height;
    if (need_buffers_ || w != desc_.width || h != desc_.height) {
        backend_->DestroyBuffers();
        FgCompBackendInfo bi;
        const FgWinResult r = backend_->CreateBuffers(info_.slot_count, w, h, &bi);
        if (!r.Ok() || FAILED(backend_->SetSourceRect(w, h))) {
            engine_.MarkFailed();  // the route is unusable at the new size: refuse, never present stale buffers
            worker_.Wake();
            return engine_.Refuse(FgStatus::ResizeFailed);
        }
        info_.d3d11_misc_flags = bi.d3d11_misc_flags;
        info_.simultaneous_access = bi.simultaneous_access;
        need_buffers_ = false;
        desc_.width = w;
        desc_.height = h;
    }
    resize_width_ = 0;
    resize_height_ = 0;
    have_vblank_ = false;
    next_pace_wake_ = 0;
    deferred_.clear();
    ForgetSubmittedLocked();
    worker_.Wake();
    return s;
}

FgStatus FgWinCompPresenter::OnDeviceLost() {
    std::lock_guard<std::mutex> lock(mutex_);
    const FgStatus s = engine_.OnDeviceLost();
    deferred_.clear();
    next_pace_wake_ = 0;
    ForgetSubmittedLocked();
    return s;
}

FgStatus FgWinCompPresenter::Detach() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint64_t from = 0;
    if (engine_.State() == FgPresenterState::Attached) {
        engine_.Quiesce(NowLocked(), &from);
        CancelOnDisplayLocked(from);
    }
    // display-held buffers do not hold up a detach: they go to the compositor and a later Attach makes new ones
    if (engine_.State() == FgPresenterState::Quiescing && !engine_.Drained() && engine_.OnlyDisplayHeld()) {
        backend_->DestroyBuffers();
        engine_.ForgetDisplayHeld();
        need_buffers_ = true;
    }
    const FgStatus s = engine_.Detach(NowLocked(), &from);
    CancelOnDisplayLocked(from);
    if (s == FgStatus::Ok) {
        if (claimed_) FgChainRegistry::Process().Release(chain_, this);
        claimed_ = false;
        deferred_.clear();
        next_pace_wake_ = 0;
        ForgetSubmittedLocked();
    }
    worker_.Wake();  // drain retired availability handles (or wake producer retry on Busy)
    return s;
}

bool FgWinCompPresenter::WaitIdle(DWORD timeout_ms) {
    const ULONGLONG until = GetTickCount64() + timeout_ms;
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!start_worker_) StepLocked();
            const FgPresenterState st = engine_.State();
            if (engine_.Drained() || st == FgPresenterState::Lost ||
                (st != FgPresenterState::Attached && engine_.OnlyDisplayHeld()))
                return true;
        }
        if (GetTickCount64() >= until) return false;
        worker_.Wake();
        Sleep(1);
    }
}

// --- display path (worker, under mutex_) -------------------------------------------------------------------------

void FgWinCompPresenter::UpdateGridLocked() {
    grid_updated_at_ = NowLocked();
    if (desc_.variable_refresh) {
        info_.grid = FgRefreshGrid();
        return;
    }
    // The compositor's last completed frame is a vblank on the scan-out grid; the caller's exact period (from the
    // display mode) wins when given. A failed read keeps the previous grid.
    FgRefreshGrid grid;
    if (backend_->ReadRefreshGrid(desc_.refresh_period, &grid)) info_.grid = grid;
}

bool FgWinCompPresenter::TakePlatformPresent(std::uint64_t platform_id, Submitted* out) {
    for (std::size_t i = 0; i < submitted_.size(); ++i) {
        if (submitted_[i].platform_id != platform_id) continue;
        *out = submitted_[i];
        submitted_.erase(submitted_.begin() + static_cast<std::ptrdiff_t>(i));
        return true;
    }
    return false;
}

void FgWinCompPresenter::LoseLocked() {
    engine_.OnDeviceLost();
    deferred_.clear();
    next_pace_wake_ = 0;
    ForgetSubmittedLocked();
}

void FgWinCompPresenter::ForgetSubmittedLocked() {
    display_.Unresolved(submitted_.size());
    display_.BreakTiming();
    submitted_.clear();
}

void FgWinCompPresenter::CancelOnDisplayLocked(std::uint64_t engine_from) {
    if (engine_from == 0) return;
    // never submitted: just forget them
    std::size_t keep = 0;
    for (std::size_t i = 0; i < deferred_.size(); ++i)
        if (deferred_[i].present_id < engine_from) deferred_[keep++] = deferred_[i];
    deferred_.resize(keep);
    if (deferred_.empty()) next_pace_wake_ = 0;
    // submitted: platform ids follow engine ids (one present per outbox entry, in order)
    std::uint64_t first = 0;
    keep = 0;
    for (std::size_t i = 0; i < submitted_.size(); ++i) {
        if (submitted_[i].engine_id >= engine_from) {
            if (first == 0 || submitted_[i].platform_id < first) first = submitted_[i].platform_id;
        } else {
            submitted_[keep++] = submitted_[i];
        }
    }
    display_.Unresolved(submitted_.size() - keep);
    submitted_.resize(keep);
    if (first != 0 && backend_) {
        const HRESULT hr = backend_->CancelPresentsFrom(first);
        if (FAILED(hr) && FgIsDeviceLostHr(hr)) LoseLocked();
    }
    // The next present continues after the last one still on its way, avoiding target collisions with uncancelled work.
    if (!submitted_.empty()) {
        have_vblank_ = true;
        last_vblank_ = submitted_.back().vblank;
    } else {
        have_vblank_ = false;
    }
}

void FgWinCompPresenter::DrainStatisticsLocked() {
    const std::uint64_t prime_pid = backend_->PrimePresentId();
    for (int guard = 0; guard < 1024; ++guard) {
        FgCompStatistic st;
        if (!backend_->PopStatistic(&st)) break;
        const std::uint64_t pid = st.present_id;
        if (prime_pid != 0 && pid == prime_pid) continue;
        Submitted submitted{};
        switch (st.kind) {
            case FgCompStatKind::PresentStatus: {
                if (st.comp_frame_id != 0) {
                    ++info_.status_frame_nonzero;
                    if (info_.status_first_cfid == 0) info_.status_first_cfid = st.comp_frame_id;
                    if (st.status_target_count > info_.status_target_count_max)
                        info_.status_target_count_max = st.status_target_count;
                    if (st.status_target_presented_time != 0) ++info_.status_target_presented_nonzero;
                    if (st.status_target_completed_time != 0) ++info_.status_target_completed_nonzero;
                }
                if (st.status == FgCompPresentStatus::Queued) {
                    ++info_.status_queued;
                    if (st.comp_frame_id != 0) {
                        ++info_.status_queued_with_cfid;
                        for (Submitted& s : submitted_) {
                            if (s.platform_id == pid) {
                                s.comp_frame_id = st.comp_frame_id;
                                s.queued_confirmed = true;
                                break;
                            }
                        }
                    }
                } else if (st.status == FgCompPresentStatus::Skipped || st.status == FgCompPresentStatus::Canceled) {
                    if (st.status == FgCompPresentStatus::Skipped) ++info_.status_skipped;
                    else ++info_.status_canceled;
                    if (TakePlatformPresent(pid, &submitted)) {
                        if (submitted.watchdog_cancelled) ++info_.watchdog_retired;
                        engine_.OnPresentSkipped(submitted.engine_id);
                        display_.Unseen(submitted.display);
                    }
                }
                break;
            }
            case FgCompStatKind::IndependentFlipFrame: {
                ++info_.indep_flip_frames;
                if (pid > last_displayed_platform_) last_displayed_platform_ = pid;
                const bool measured = st.displayed_measured && st.displayed_time > 0;
                const FgTicks actual = measured ? st.displayed_time : NowLocked();
                if (TakePlatformPresent(pid, &submitted)) {
                    if (measured) ++info_.display_time_iflip;
                    else ++info_.display_time_estimated;
                    engine_.OnPresentDisplayed(submitted.engine_id, actual, measured);
                    display_.Displayed(submitted.display, measured ? actual : 0);
                }
                break;
            }
            case FgCompStatKind::CompositionFrame: {
                ++info_.comp_frames;
                if (pid > last_displayed_platform_) last_displayed_platform_ = pid;
                const bool measured = st.displayed_measured && st.displayed_time > 0;
                const FgTicks actual =
                    measured ? st.displayed_time : (st.displayed_time > 0 ? st.displayed_time : NowLocked());
                if (TakePlatformPresent(pid, &submitted)) {
                    if (measured) ++info_.display_time_comp;
                    else ++info_.display_time_estimated;
                    engine_.OnPresentDisplayed(submitted.engine_id, actual, measured);
                    display_.Displayed(submitted.display, measured ? actual : 0);
                }
                break;
            }
            case FgCompStatKind::None:
                break;
        }
    }
}

void FgWinCompPresenter::ResolveQueuedDisplaysLocked() {
    if (submitted_.empty()) return;
    const std::uint64_t completed_cfid = backend_->CompletedCompositionFrameId();
    const std::uint64_t retire_val = backend_->RetiringFenceValue();
    info_.retiring_fence_value = retire_val;
    if (completed_cfid == 0 && retire_val == 0) return;

    const FgTicks now = NowLocked();
    const FgTicks dcomp_grace = std::max<FgTicks>(RefreshPeriodLocked() * 4, 400000);  // >= 40 ms
    std::size_t i = 0;
    while (i < submitted_.size()) {
        const Submitted& s = submitted_[i];
        if (!s.queued_confirmed || s.comp_frame_id == 0) {
            ++i;
            continue;
        }
        const bool cfid_done = completed_cfid != 0 && completed_cfid >= s.comp_frame_id;
        const bool fence_done = retire_val != 0 && retire_val >= s.platform_id;
        if (!cfid_done && !fence_done) {
            ++i;
            continue;
        }
        // When DCompositionGetFrameId(COMPLETED) is available but has not yet crossed s.comp_frame_id, wait briefly
        // before falling back to retiring-fence confirmation so we retrieve the measured DComp target timestamp.
        if (!cfid_done && completed_cfid != 0 && s.vblank > 0 && now < s.vblank + dcomp_grace) {
            ++i;
            continue;
        }

        FgCompFrameTiming timing;
        const bool probed = backend_->ProbeCompFrameTiming(s.comp_frame_id, &timing);
        if (probed) {
            if (timing.target_count > info_.status_target_count_max)
                info_.status_target_count_max = timing.target_count;
            if (timing.raw_presented_qpc != 0) ++info_.status_target_presented_nonzero;
            if (timing.raw_completed_qpc != 0) ++info_.status_target_completed_nonzero;
        }

        bool measured = false;
        FgTicks actual = 0;
        if (probed && timing.completed_time > 0) {
            actual = timing.completed_time;
            measured = true;
            ++info_.display_time_dcomp_stats;
        } else if (probed && timing.presented_time > 0) {
            actual = timing.presented_time;
            measured = true;
            ++info_.display_time_dcomp_stats;
        } else {
            actual = (probed && timing.target_present_time > 0) ? timing.target_present_time : now;
            measured = false;
            ++info_.display_time_estimated;
        }

        const Submitted done = s;
        submitted_.erase(submitted_.begin() + static_cast<std::ptrdiff_t>(i));
        if (done.platform_id > last_displayed_platform_) last_displayed_platform_ = done.platform_id;
        engine_.OnPresentDisplayed(done.engine_id, actual, measured);
        display_.Displayed(done.display, measured ? actual : 0);
    }
}

void FgWinCompPresenter::ReleaseSlotsLocked() {
    const FgTicks now = NowLocked();
    scratch_slots_.clear();
    engine_.SlotsAwaitingRelease(&scratch_slots_);
    for (const std::uint32_t slot : scratch_slots_) {
        if (slot >= info_.slot_count) continue;
        if (engine_.SlotState(slot) == FgEngineSlotState::Queued) {
            const std::uint64_t eid = engine_.SlotPresentId(slot);
            for (std::size_t i = 0; i < submitted_.size(); ++i) {
                if (submitted_[i].engine_id != eid) continue;
                const bool older_available = !submitted_[i].queued_confirmed && last_displayed_platform_ != 0 &&
                                             submitted_[i].platform_id < last_displayed_platform_;
                // Windowed-only 250 ms occlusion watchdog (never triggers in headless bind_window==false):
                // if DWM stops progressing the queue while occluded/minimized, cancel the stalled suffix and retire
                // a queued present only once IsBufferAvailable confirms release.
                const bool occluded_timeout = desc_.bind_window && submitted_[i].vblank > 0 &&
                                              now > submitted_[i].vblank + kOcclusionWatchdogTicks;
                if (older_available || occluded_timeout) {
                    if (backend_->IsBufferAvailable(slot)) {
                        if (occluded_timeout) ++info_.watchdog_retired;
                        display_.Unseen(submitted_[i].display);
                        submitted_.erase(submitted_.begin() + static_cast<std::ptrdiff_t>(i));
                        engine_.OnPresentSkipped(eid);
                    } else if (occluded_timeout && !submitted_[i].watchdog_cancelled) {
                        const std::uint64_t cancel_pid = submitted_[i].platform_id;
                        for (std::size_t j = i; j < submitted_.size(); ++j) submitted_[j].watchdog_cancelled = true;
                        const HRESULT hr = backend_->CancelPresentsFrom(cancel_pid);
                        if (FAILED(hr) && FgIsDeviceLostHr(hr)) {
                            LoseLocked();
                            return;
                        }
                    }
                }
                break;
            }
        }
        if (engine_.SlotState(slot) != FgEngineSlotState::Retired) continue;
        if (backend_->IsBufferAvailable(slot)) engine_.OnSlotReleased(slot, false);
    }
}

void FgWinCompPresenter::SubmitOutboxLocked() {
    engine_.TakeOutbox(&deferred_);
    next_pace_wake_ = 0;
    const FgTicks now = NowLocked();
    std::size_t i = 0;
    for (; i < deferred_.size(); ++i) {
        const FgEnginePresent& p = deferred_[i];
        if (engine_.SlotPresentId(p.slot) != p.present_id || p.slot >= info_.slot_count) continue;  // cancelled since
        if (p.represent && !backend_->IsBufferAvailable(p.slot)) break;  // cancelled present must release buffer first

        const bool real_only_plan = p.real && p.index == 1;
        const std::size_t max_in_flight = real_only_plan ? kMaxRealOnlyInFlight : kMaxCompInFlight;
        if (submitted_.size() >= max_in_flight) break;

        FgTicks vblank = p.target;
        FgTicks when = p.target;
        if (!desc_.variable_refresh) {
            if (FgSnapToRefreshGrid(p.target, info_.grid, &vblank, &when) != FgStatus::Ok) {
                engine_.MarkFailed();
                deferred_.clear();
                return;
            }
            if (have_vblank_ && vblank <= last_vblank_) {
                // two images on one vblank would make the display skip one: a generated one yields, the real one
                // moves to the next vblank (never dropped)
                if (!p.real) {
                    engine_.OnPresentDropped(p.present_id, FgDropReason::Pressure);
                    continue;
                }
                vblank = last_vblank_ + info_.grid.period;
                when = vblank - info_.grid.period / 2;
            }
        } else {
            const FgTicks min_gap =
                desc_.min_interval > 0 ? desc_.min_interval : (desc_.refresh_period > 0 ? desc_.refresh_period : 1);
            const FgTicks next_target = i + 1 < deferred_.size() ? deferred_[i + 1].target : engine_.HeadPendingTarget();
            const FgTicks earliest = have_vblank_ ? last_vblank_ + min_gap : 0;
            const bool delayed = have_vblank_ && earliest > p.target;
            const FgTearingPlanDecision td = FgPlanTearingImage(
                p.target, p.real, now, earliest, min_gap, 20000,
                next_target > p.target, next_target, delayed);
            if (td.DropLate()) {
                engine_.OnPresentDropped(p.present_id, FgDropReason::Late);
                continue;
            }
            FgTicks effective = td.issue_at > now ? td.issue_at : now;
            if (have_vblank_ && effective < last_vblank_ + min_gap) effective = last_vblank_ + min_gap;
            when = vblank = effective;
        }

        // Pace submission so real-only startup loops never flood IPresentationManager and composed-mode intervals
        // arrive just before each image's target time instead of bursting all N+1 frames into one composition frame.
        const FgTicks lookahead = PresentLookaheadLocked(real_only_plan);
        if (lookahead > 0 && now + lookahead < when) {
            next_pace_wake_ = when - lookahead;
            break;
        }

        std::uint64_t pid = 0;
        const HRESULT hr = backend_->PresentSlot(p.slot, when, p.fence, &pid);
        if (FAILED(hr)) {
            info_.last_present_hr = hr;
            if (FgIsDeviceLostHr(hr)) {
                LoseLocked();
            } else {
                engine_.MarkFailed();  // a failed present is a hard failure: refuse until Detach, no fallback
                deferred_.clear();
                ForgetSubmittedLocked();
            }
            return;
        }
        const FgTicks submitted_at = NowLocked();
        const bool visible = backend_->IsWindowVisible();
        info_.window_visible = visible;
        if (visible) {
            ++info_.visible_submitted;
            if (info_.first_visible_submit == 0) info_.first_visible_submit = submitted_at;
            info_.last_visible_submit = submitted_at;
        }
        engine_.OnPresentSubmitted(p.present_id, submitted_at);
        display_.Submitted();
        submitted_.push_back(
            Submitted{p.present_id, pid, vblank, {p.target, submitted_at, p.real, p.gameplay}, 0, false, false});
        if (submitted_.size() > kMaxSubmitted) {
            display_.Unresolved(1);
            submitted_.erase(submitted_.begin());
        }
        have_vblank_ = true;
        last_vblank_ = vblank;
    }
    deferred_.erase(deferred_.begin(), deferred_.begin() + static_cast<std::ptrdiff_t>(i));
}

void FgWinCompPresenter::StepLocked() {
    const FgPresenterState st = engine_.State();
    if (st == FgPresenterState::Lost || st == FgPresenterState::Detached) return;
    if (backend_->IsLost()) {
        LoseLocked();
        return;
    }
    const FgTicks now = NowLocked();
    if (!desc_.variable_refresh && now - grid_updated_at_ > kGridRefresh) UpdateGridLocked();
    DrainStatisticsLocked();
    ResolveQueuedDisplaysLocked();
    ReleaseSlotsLocked();
    engine_.Pump(now);
    if (engine_.State() == FgPresenterState::Attached) SubmitOutboxLocked();
}

void FgWinCompPresenter::Run(HANDLE stop, HANDLE wake) {
    HANDLE handles[MAXIMUM_WAIT_OBJECTS] = {};
    bool throttle_available = false;
    for (;;) {
        DWORD n = 0;
        handles[n++] = stop;
        handles[n++] = wake;
        if (HANDLE h = backend_->StatsEvent()) handles[n++] = h;
        if (HANDLE h = backend_->LostEvent()) handles[n++] = h;
        if (timer_.Valid()) handles[n++] = timer_.Handle();
        if (HANDLE h = backend_->FenceEvent()) handles[n++] = h;
        const DWORD fixed = n;
        DWORD timeout = 100;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            FgTicks next_timer = 0;
            const auto consider_timer = [&](FgTicks t) {
                if (t > 0 && (next_timer == 0 || t < next_timer)) next_timer = t;
            };
            const FgTicks deadline = engine_.NextDeadline();
            if (deadline > 0) consider_timer(deadline + 1);
            if (next_pace_wake_ > 0) consider_timer(next_pace_wake_);
            bool awaiting_dcomp_completion = false;
            for (const Submitted& s : submitted_) {
                if (s.queued_confirmed && s.comp_frame_id != 0) {
                    awaiting_dcomp_completion = true;
                    break;
                }
            }
            if (awaiting_dcomp_completion) {
                const FgTicks poll_step = std::min<FgTicks>(RefreshPeriodLocked(), 30000);
                consider_timer(NowLocked() + std::max<FgTicks>(poll_step, 10000));
            }
            if (desc_.bind_window && !submitted_.empty() && submitted_.front().vblank > 0 &&
                !submitted_.front().watchdog_cancelled) {
                consider_timer(submitted_.front().vblank + kOcclusionWatchdogTicks + 1);
            }
            if (next_timer > 0) timer_.ArmAt(next_timer);
            else timer_.Cancel();

            FgReadyFence head;
            if (engine_.HeadPendingFence(&head)) backend_->WatchFence(head);
            if (engine_.PendingConsumptionFence(&head)) backend_->WatchFence(head);

            if (throttle_available) {
                timeout = 2;
            } else {
                scratch_slots_.clear();
                engine_.SlotsAwaitingRelease(&scratch_slots_);
                for (const std::uint32_t slot : scratch_slots_) {
                    const FgEngineSlotState st = engine_.SlotState(slot);
                    bool watch = st == FgEngineSlotState::Retired;
                    if (!watch && st == FgEngineSlotState::Queued) {
                        const std::uint64_t eid = engine_.SlotPresentId(slot);
                        for (const Submitted& s : submitted_) {
                            if (s.engine_id == eid && s.watchdog_cancelled) {
                                watch = true;
                                break;
                            }
                        }
                    }
                    if (watch && n < MAXIMUM_WAIT_OBJECTS) {
                        if (HANDLE avail = backend_->SlotAvailableEvent(slot)) handles[n++] = avail;
                    }
                }
                for (const FgEnginePresent& p : deferred_) {
                    if (p.represent && n < MAXIMUM_WAIT_OBJECTS) {
                        if (HANDLE avail = backend_->SlotAvailableEvent(p.slot)) handles[n++] = avail;
                    }
                }
            }
        }
        const DWORD w = WaitForMultipleObjects(n, handles, FALSE, timeout);
        if (w == WAIT_OBJECT_0) break;
        std::lock_guard<std::mutex> lock(mutex_);
        backend_->DrainRetiredHandles();
        const std::uint32_t free_before = engine_.FreeSlots();
        const std::size_t deferred_before = deferred_.size();
        const std::size_t submitted_before = submitted_.size();
        StepLocked();
        // an availability wake that released nothing (the event can stay signalled while the slot waits for its
        // statistic): poll briefly instead of spinning on it
        throttle_available = w >= WAIT_OBJECT_0 + fixed && w < WAIT_OBJECT_0 + n &&
                             engine_.FreeSlots() == free_before && deferred_.size() == deferred_before &&
                             submitted_.size() == submitted_before;
    }
    timer_.Cancel();
}

}  // namespace wgfg
