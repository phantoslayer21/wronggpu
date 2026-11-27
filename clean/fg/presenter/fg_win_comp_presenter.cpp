// SPDX-License-Identifier: GPL-3.0-only
// Windows composition presenter (presentation manager + hardware flip queue): see fg_win_comp_presenter.h.
#include "fg_win_comp_presenter.h"

#include <cstddef>

namespace wgfg {

using Microsoft::WRL::ComPtr;

namespace {

constexpr std::uint32_t kMaxSlots = 32;
constexpr FgTicks kGridRefresh = 10000000;  // re-read the compositor's vblank phase once a second
constexpr std::size_t kMaxSubmitted = 256;

bool ScanoutFormat(DXGI_FORMAT f) {
    return f == DXGI_FORMAT_B8G8R8A8_UNORM || f == DXGI_FORMAT_R8G8B8A8_UNORM || f == DXGI_FORMAT_R10G10B10A2_UNORM ||
           f == DXGI_FORMAT_R16G16B16A16_FLOAT;
}

DXGI_COLOR_SPACE_TYPE ColorSpaceFor(DXGI_FORMAT f, bool hdr) {
    if (f == DXGI_FORMAT_R16G16B16A16_FLOAT) return DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
    if (f == DXGI_FORMAT_R10G10B10A2_UNORM && hdr) return DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    return DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
}

}  // namespace

FgWinCompPresenter::FgWinCompPresenter(const FgCompPresenterDesc& desc, const FgEngineConfig& engine)
    : desc_(desc), device12_(desc.device), engine_(engine, &fences12_) {
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

    FgEngineConfig cfg;
    cfg.probe = FgPresenterProbe{FgPresenterBackend::WindowsComposition, true, true, d.max_generated, FgStatus::Ok};
    cfg.slot_count = slots;
    cfg.max_pending_intervals = d.max_pending_intervals >= 1 ? d.max_pending_intervals : 2u;
    cfg.refresh_period = d.refresh_period;
    std::unique_ptr<FgWinCompPresenter> p(new FgWinCompPresenter(d, cfg));
    const FgWinResult r = p->Initialize();
    if (!r.Ok()) return r;
    *out = std::move(p);
    return FgWinOk();
}

FgWinResult FgWinCompPresenter::Initialize() {
    if (!timer_.Valid()) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::TimerUnavailable);
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event_) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation);

    const LUID luid = device12_->GetAdapterLuid();
    const FgWinResult caps = FgQueryAdapterFlipCaps(luid, &info_.adapter);
    if (desc_.require_hw_flip_queue && (!caps.Ok() || !info_.adapter.queried || !info_.adapter.hw_flip_queue_enabled))
        return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NoHardwareFlipQueue, info_.adapter.status);

    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterNotFound, hr);
    ComPtr<IDXGIAdapter1> adapter;
    hr = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
    if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterNotFound, hr);

    // The presentation device: Direct3D 11 on the same adapter. Not single-threaded: creation and the worker use it
    // (always under mutex_).
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels,
                           2, D3D11_SDK_VERSION, &dev, nullptr, &ctx);
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

    FgWinResult r = CreateBuffers(desc_.width, desc_.height);
    if (!r.Ok()) return r;

    hr = FgDCompCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, nullptr, &surface_handle_);
    if (FAILED(hr)) return FgWinFail(FgStatus::Refused, FgWinRefusal::CompositionTarget, hr);
    hr = manager_->CreatePresentationSurface(surface_handle_, &surface_);
    if (FAILED(hr)) return FgWinFail(FgStatus::Refused, FgWinRefusal::CompositionTarget, hr);
    surface_->SetTag(1);
    const RECT src = {0, 0, static_cast<LONG>(desc_.width), static_cast<LONG>(desc_.height)};
    if (FAILED(hr = surface_->SetAlphaMode(DXGI_ALPHA_MODE_IGNORE)) ||
        FAILED(hr = surface_->SetColorSpace(ColorSpaceFor(desc_.format, desc_.hdr))) ||
        FAILED(hr = surface_->SetSourceRect(&src)))
        return FgWinFail(FgStatus::Refused, FgWinRefusal::CompositionTarget, hr);

    if (desc_.bind_window) {
        if (FAILED(hr = FgDCompCreateDevice3(nullptr, IID_PPV_ARGS(&dcomp_))) ||
            FAILED(hr = dcomp_->CreateTargetForHwnd(desc_.hwnd, TRUE, &target_)) ||
            FAILED(hr = dcomp_->CreateVisual(&visual_)) ||
            FAILED(hr = dcomp_->CreateSurfaceFromHandle(surface_handle_, &dcomp_surface_)) ||
            FAILED(hr = visual_->SetContent(dcomp_surface_.Get())) || FAILED(hr = target_->SetRoot(visual_.Get())) ||
            FAILED(hr = dcomp_->Commit()))
            return FgWinFail(FgStatus::Refused, FgWinRefusal::CompositionTarget, hr);
    }

    hr = manager_->ForceVSyncInterrupt(desc_.force_vsync_interrupt ? 1 : 0);
    if (FAILED(hr)) return FgWinFail(FgStatus::Refused, FgWinRefusal::StatisticsUnavailable, hr);
    if (FAILED(hr = manager_->EnablePresentStatisticsKind(PresentStatisticsKind_PresentStatus, 1)) ||
        FAILED(hr = manager_->EnablePresentStatisticsKind(PresentStatisticsKind_CompositionFrame, 1)) ||
        FAILED(hr = manager_->EnablePresentStatisticsKind(PresentStatisticsKind_IndependentFlipFrame, 1)) ||
        FAILED(hr = manager_->GetPresentStatisticsAvailableEvent(&stats_event_)) ||
        FAILED(hr = manager_->GetLostEvent(&lost_event_)))
        return FgWinFail(FgStatus::Refused, FgWinRefusal::StatisticsUnavailable, hr);

    if (!desc_.variable_refresh) {
        UpdateGridLocked();  // no worker yet: nothing to lock against
        if (info_.grid.period <= 0) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::RefreshUnknown);
    }
    if (!worker_.Start(this, L"wg-fg-composition-presenter", desc_.worker_registration))
        return FgWinFail(FgStatus::Refused, FgWinRefusal::WorkerStart);
    return FgWinOk();
}

FgWinResult FgWinCompPresenter::CreateBuffers(UINT width, UINT height) {
    buffers_.resize(info_.slot_count);
    for (std::uint32_t i = 0; i < info_.slot_count; ++i) {
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
        HRESULT hr = device12_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_DISPLAY, &rd,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
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
    return FgWinOk();
}

void FgWinCompPresenter::DestroyBuffers() {
    for (Buffer& b : buffers_) {
        if (b.available) CloseHandle(b.available);
        b = Buffer();
    }
    buffers_.clear();
}

FgWinCompPresenter::~FgWinCompPresenter() {
    worker_.Stop();
    std::lock_guard<std::mutex> lock(mutex_);
    if (claimed_) FgChainRegistry::Process().Release(chain_, this);
    claimed_ = false;
    if (manager_ && !submitted_.empty()) manager_->CancelPresentsFrom(submitted_.front().platform_id);
    DestroyBuffers();
    if (stats_event_) CloseHandle(stats_event_);
    if (lost_event_) CloseHandle(lost_event_);
    if (fence_event_) CloseHandle(fence_event_);
    if (surface_handle_) CloseHandle(surface_handle_);
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
    std::lock_guard<std::mutex> lock(mutex_);
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

ID3D12Resource* FgWinCompPresenter::SlotResource(std::uint32_t slot) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slot < buffers_.size() ? buffers_[slot].resource.Get() : nullptr;
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
    return info_;
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
        DestroyBuffers();
        if (!CreateBuffers(desc_.width, desc_.height).Ok()) return engine_.Refuse(FgStatus::Refused);
        need_buffers_ = false;
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
    return engine_.BeginInterval(plan, FgInterruptNow(), out);
}

FgStatus FgWinCompPresenter::SubmitInterval(const FgIntervalSubmission& submission) {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        s = engine_.SubmitInterval(submission, FgInterruptNow());
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
        s = engine_.RevokeGenerated(interval_id, reason, FgInterruptNow(), &from);
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
        n = engine_.CancelPresentsFrom(present_id, FgInterruptNow(), &from);
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
        s = engine_.Quiesce(FgInterruptNow(), &from);
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
        DestroyBuffers();
        engine_.ForgetDisplayHeld();
        need_buffers_ = true;
    }
    const FgStatus s = engine_.CompleteResize(resize_ok, new_epoch);
    if (s != FgStatus::Ok) return s;
    const UINT w = resize_width_ ? resize_width_ : desc_.width;
    const UINT h = resize_height_ ? resize_height_ : desc_.height;
    if (need_buffers_ || w != desc_.width || h != desc_.height) {
        DestroyBuffers();
        const FgWinResult r = CreateBuffers(w, h);
        const RECT src = {0, 0, static_cast<LONG>(w), static_cast<LONG>(h)};
        if (!r.Ok() || FAILED(surface_->SetSourceRect(&src))) {
            engine_.MarkFailed();  // the route is unusable at the new size: refuse, never present stale buffers
            return engine_.Refuse(FgStatus::ResizeFailed);
        }
        need_buffers_ = false;
        desc_.width = w;
        desc_.height = h;
    }
    resize_width_ = 0;
    resize_height_ = 0;
    have_vblank_ = false;
    deferred_.clear();
    submitted_.clear();
    return s;
}

FgStatus FgWinCompPresenter::OnDeviceLost() {
    std::lock_guard<std::mutex> lock(mutex_);
    const FgStatus s = engine_.OnDeviceLost();
    deferred_.clear();
    submitted_.clear();
    return s;
}

FgStatus FgWinCompPresenter::Detach() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint64_t from = 0;
    if (engine_.State() == FgPresenterState::Attached) {
        engine_.Quiesce(FgInterruptNow(), &from);
        CancelOnDisplayLocked(from);
    }
    // display-held buffers do not hold up a detach: they go to the compositor and a later Attach makes new ones
    if (engine_.State() == FgPresenterState::Quiescing && !engine_.Drained() && engine_.OnlyDisplayHeld()) {
        DestroyBuffers();
        engine_.ForgetDisplayHeld();
        need_buffers_ = true;
    }
    const FgStatus s = engine_.Detach(FgInterruptNow(), &from);
    CancelOnDisplayLocked(from);
    if (s == FgStatus::Ok) {
        if (claimed_) FgChainRegistry::Process().Release(chain_, this);
        claimed_ = false;
        deferred_.clear();
        submitted_.clear();
    } else {
        worker_.Wake();  // producer leases still held: the caller retries (WaitIdle)
    }
    return s;
}

bool FgWinCompPresenter::WaitIdle(DWORD timeout_ms) {
    const ULONGLONG until = GetTickCount64() + timeout_ms;
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
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
    grid_updated_at_ = FgInterruptNow();
    if (desc_.variable_refresh) {
        info_.grid = FgRefreshGrid();
        return;
    }
    // The compositor's last completed frame is a vblank on the scan-out grid; the caller's exact period (from the
    // display mode) wins when given. A failed read keeps the previous grid.
    FgRefreshGrid grid;
    if (FgReadCompositorGrid(clock_map_, desc_.refresh_period, &grid)) info_.grid = grid;
}

bool FgWinCompPresenter::EngineIdForPlatform(std::uint64_t platform_id, std::uint64_t* engine_id, bool erase) {
    for (std::size_t i = 0; i < submitted_.size(); ++i) {
        if (submitted_[i].platform_id != platform_id) continue;
        *engine_id = submitted_[i].engine_id;
        if (erase) submitted_.erase(submitted_.begin() + static_cast<std::ptrdiff_t>(i));
        return true;
    }
    return false;
}

void FgWinCompPresenter::LoseLocked() {
    engine_.OnDeviceLost();
    deferred_.clear();
    submitted_.clear();
}

void FgWinCompPresenter::CancelOnDisplayLocked(std::uint64_t engine_from) {
    if (engine_from == 0) return;
    // never submitted: just forget them
    std::size_t keep = 0;
    for (std::size_t i = 0; i < deferred_.size(); ++i)
        if (deferred_[i].present_id < engine_from) deferred_[keep++] = deferred_[i];
    deferred_.resize(keep);
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
    submitted_.resize(keep);
    if (first != 0 && manager_) {
        const HRESULT hr = manager_->CancelPresentsFrom(first);
        if (FAILED(hr) && FgIsDeviceLostHr(hr)) LoseLocked();
    }
    // the next present continues after the last one still on its way, not after the cancelled ones
    have_vblank_ = false;
}

void FgWinCompPresenter::DrainStatisticsLocked() {
    for (int guard = 0; guard < 1024; ++guard) {
        ComPtr<IPresentStatistics> st;
        if (manager_->GetNextPresentStatistics(&st) != S_OK || !st) break;
        const std::uint64_t pid = st->GetPresentId();
        std::uint64_t eid = 0;
        switch (st->GetKind()) {
            case PresentStatisticsKind_PresentStatus: {
                ComPtr<IPresentStatusPresentStatistics> ps;
                if (FAILED(st.As(&ps))) break;
                const PresentStatus v = ps->GetPresentStatus();
                if ((v == PresentStatus_Skipped || v == PresentStatus_Canceled) && EngineIdForPlatform(pid, &eid, true))
                    engine_.OnPresentSkipped(eid);
                break;
            }
            case PresentStatisticsKind_IndependentFlipFrame: {
                ComPtr<IIndependentFlipFramePresentStatistics> fs;
                if (FAILED(st.As(&fs))) break;
                ++info_.indep_flip_frames;
                if (pid > last_displayed_platform_) last_displayed_platform_ = pid;
                const FgTicks timestamp = static_cast<FgTicks>(fs->GetDisplayedTime().value);
                if (EngineIdForPlatform(pid, &eid, true))
                    engine_.OnPresentDisplayed(eid, timestamp != 0 ? timestamp : FgInterruptNow(), timestamp != 0);
                break;
            }
            case PresentStatisticsKind_CompositionFrame: {
                ComPtr<ICompositionFramePresentStatistics> cs;
                ++info_.comp_frames;
                FgTicks displayed_at = FgInterruptNow();
                bool measured = false;
                if (SUCCEEDED(st.As(&cs))) {
                    const COMPOSITION_FRAME_ID fid = cs->GetCompositionFrameId();
                    COMPOSITION_FRAME_STATS fst = {};
                    COMPOSITION_TARGET_ID targets[4] = {};
                    UINT count = 0;
                    COMPOSITION_TARGET_STATS ts = {};
                    if (SUCCEEDED(FgDCompGetStatistics(fid, &fst, 4, targets, &count)) && count > 0 &&
                        SUCCEEDED(FgDCompGetTargetStatistics(fid, &targets[0], &ts)) && ts.presentTime != 0) {
                        displayed_at = clock_map_.FromQpc(static_cast<std::int64_t>(ts.presentTime));
                        measured = true;
                    }
                }
                if (pid > last_displayed_platform_) last_displayed_platform_ = pid;
                if (EngineIdForPlatform(pid, &eid, true)) engine_.OnPresentDisplayed(eid, displayed_at, measured);
                break;
            }
            default: break;
        }
    }
}

void FgWinCompPresenter::ReleaseSlotsLocked() {
    const FgTicks now = FgInterruptNow();
    scratch_slots_.clear();
    engine_.SlotsAwaitingRelease(&scratch_slots_);
    for (const std::uint32_t slot : scratch_slots_) {
        if (slot >= buffers_.size()) continue;
        if (engine_.SlotState(slot) == FgEngineSlotState::Queued) {
            const std::uint64_t eid = engine_.SlotPresentId(slot);
            for (std::size_t i = 0; i < submitted_.size(); ++i) {
                if (submitted_[i].engine_id != eid) continue;
                const bool older_available =
                    last_displayed_platform_ != 0 && submitted_[i].platform_id < last_displayed_platform_;
                // A2 occlusion watchdog (windowed only; never triggers in headless bind_window==false validation):
                // if DWM stops emitting statistics while occluded/minimized, retire stale queued presents after 250 ms
                // so the game's frame-latency credit contract never deadlocks.
                const bool occluded_timeout = desc_.bind_window && submitted_[i].target > 0 &&
                                              now > submitted_[i].target + 2500000;
                boolean available = 0;
                if ((older_available || occluded_timeout) &&
                    SUCCEEDED(buffers_[slot].buffer->IsAvailable(&available)) && available) {
                    submitted_.erase(submitted_.begin() + static_cast<std::ptrdiff_t>(i));
                    engine_.OnPresentSkipped(eid);
                }
                break;
            }
        }
        if (engine_.SlotState(slot) != FgEngineSlotState::Retired) continue;
        boolean available = 0;
        if (SUCCEEDED(buffers_[slot].buffer->IsAvailable(&available)) && available)
            engine_.OnSlotReleased(slot, false);
    }
}

void FgWinCompPresenter::SubmitOutboxLocked() {
    engine_.TakeOutbox(&deferred_);
    const FgTicks now = FgInterruptNow();
    std::size_t i = 0;
    for (; i < deferred_.size(); ++i) {
        const FgEnginePresent& p = deferred_[i];
        if (engine_.SlotPresentId(p.slot) != p.present_id || p.slot >= buffers_.size()) continue;  // cancelled since
        Buffer& b = buffers_[p.slot];
        if (p.represent) {  // the cancelled present of this buffer must be gone first
            boolean available = 0;
            if (FAILED(b.buffer->IsAvailable(&available)) || !available) break;
        }
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
            // A3: avoid HWFQ VRR target collisions / compositor skips when an earlier generated image is already
            // superseded by the next queued image or collides with the previous effective target.
            const FgTicks gap = desc_.min_interval > 2500 ? desc_.min_interval : 2500;
            FgTicks effective = when > now ? when : now;
            if (have_vblank_ && effective < last_vblank_ + desc_.min_interval)
                effective = last_vblank_ + desc_.min_interval;
            if (!p.real && i + 1 < deferred_.size() && effective > deferred_[i + 1].target - gap) {
                engine_.OnPresentDropped(p.present_id, FgDropReason::Late);
                continue;
            }
            if (have_vblank_ && effective <= last_vblank_) effective = last_vblank_ + 1;
            when = vblank = effective;
        }
        // GPU-side order: the presentation device waits for the producer fence before the present (already complete
        // on the CPU; this keeps the order explicit on the GPU timeline too).
        if (p.fence.fence >= 1 && p.fence.fence <= fences11_.size() && fences11_[p.fence.fence - 1])
            context11_->Wait(fences11_[p.fence.fence - 1].Get(), p.fence.value);
        HRESULT hr = surface_->SetBuffer(b.buffer.Get());
        if (SUCCEEDED(hr)) hr = manager_->SetTargetTime(SystemInterruptTime{static_cast<UINT64>(when)});
        const std::uint64_t pid = manager_->GetNextPresentId();
        if (SUCCEEDED(hr)) hr = manager_->Present();
        if (FAILED(hr)) {
            if (FgIsDeviceLostHr(hr)) {
                LoseLocked();
            } else {
                engine_.MarkFailed();  // a failed present is a hard failure: refuse until Detach, no fallback
                deferred_.clear();
                submitted_.clear();
            }
            return;
        }
        engine_.OnPresentSubmitted(p.present_id, FgInterruptNow());
        submitted_.push_back(Submitted{p.present_id, pid, p.slot, vblank, p.real});
        if (submitted_.size() > kMaxSubmitted) submitted_.erase(submitted_.begin());
        have_vblank_ = true;
        last_vblank_ = vblank;
    }
    deferred_.erase(deferred_.begin(), deferred_.begin() + static_cast<std::ptrdiff_t>(i));
}

void FgWinCompPresenter::StepLocked() {
    const FgPresenterState st = engine_.State();
    if (st == FgPresenterState::Lost || st == FgPresenterState::Detached) return;
    if ((lost_event_ && WaitForSingleObject(lost_event_, 0) == WAIT_OBJECT_0) ||
        device12_->GetDeviceRemovedReason() != S_OK) {
        LoseLocked();
        return;
    }
    const FgTicks now = FgInterruptNow();
    if (!desc_.variable_refresh && now - grid_updated_at_ > kGridRefresh) UpdateGridLocked();
    DrainStatisticsLocked();
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
        handles[n++] = stats_event_;
        handles[n++] = lost_event_;
        handles[n++] = timer_.Handle();
        handles[n++] = fence_event_;
        const DWORD fixed = n;
        DWORD timeout = 100;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const FgTicks deadline = engine_.NextDeadline();
            if (deadline != 0) timer_.ArmAt(deadline + 1);
            else timer_.Cancel();
            FgReadyFence head;
            if (engine_.HeadPendingFence(&head)) {
                ID3D12Fence* f = fences12_.Get(head.fence);
                if (f) f->SetEventOnCompletion(head.value, fence_event_);
            }
            if (throttle_available) {
                timeout = 2;
            } else {
                scratch_slots_.clear();
                engine_.SlotsAwaitingRelease(&scratch_slots_);
                for (const std::uint32_t slot : scratch_slots_)
                    if (engine_.SlotState(slot) == FgEngineSlotState::Retired && slot < buffers_.size() &&
                        buffers_[slot].available && n < MAXIMUM_WAIT_OBJECTS)
                        handles[n++] = buffers_[slot].available;
                for (const FgEnginePresent& p : deferred_)
                    if (p.represent && p.slot < buffers_.size() && buffers_[p.slot].available &&
                        n < MAXIMUM_WAIT_OBJECTS)
                        handles[n++] = buffers_[p.slot].available;
            }
        }
        const DWORD w = WaitForMultipleObjects(n, handles, FALSE, timeout);
        if (w == WAIT_OBJECT_0) break;
        std::lock_guard<std::mutex> lock(mutex_);
        const std::uint32_t free_before = engine_.FreeSlots();
        const std::size_t deferred_before = deferred_.size();
        StepLocked();
        // an availability wake that released nothing (the event can stay signalled while the slot waits for its
        // statistic): poll briefly instead of spinning on it
        throttle_available = w >= WAIT_OBJECT_0 + fixed && w < WAIT_OBJECT_0 + n &&
                             engine_.FreeSlots() == free_before && deferred_.size() == deferred_before;
    }
    timer_.Cancel();
}

}  // namespace wgfg
