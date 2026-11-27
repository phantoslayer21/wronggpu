// SPDX-License-Identifier: GPL-3.0-only
// DXGI flip-discard presenter: see fg_dxgi_presenter.h.
#include "fg_dxgi_presenter.h"

#include <cstddef>

namespace wgfg {

using Microsoft::WRL::ComPtr;

namespace {

constexpr std::uint32_t kMaxSlots = 32;
constexpr FgTicks kGridRefresh = 10000000;  // re-read the vblank phase once a second
constexpr DWORD kLatencyWaitMs = 100;

}  // namespace

FgDxgiPresenter::FgDxgiPresenter(const FgDxgiPresenterDesc& desc, const FgEngineConfig& engine)
    : desc_(desc), device_(desc.device), queue_(desc.queue), swap_(desc.swap_chain), engine_(engine, &fences_) {
    info_.slot_count = engine.slot_count;
}

FgWinResult FgDxgiPresenter::Create(const FgDxgiPresenterDesc& d, std::unique_ptr<FgDxgiPresenter>* out) {
    if (!out) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::None);
    out->reset();
    // plain arguments before the objects, so every argument refusal is decided without touching a driver
    if (d.pacing != FgDxgiPacing::Tearing && d.pacing != FgDxgiPacing::Vsync)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::UnknownChoice);
    if (d.max_generated < 1 || d.max_generated > kFgMaxGenerated)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::BadCount);
    const std::uint32_t default_slots = FgMinSlotsFor(d.max_generated) < 8 ? 8u : FgMinSlotsFor(d.max_generated);
    const std::uint32_t slots = d.slot_count ? d.slot_count : default_slots;
    if (slots < FgMinSlotsFor(d.max_generated) || slots > kMaxSlots)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::SlotCount);
    if (d.refresh_period < 0 || d.late_tolerance < 0 || d.timer_lead < 0)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::RefreshUnknown);
    if (!d.device) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullDevice);
    if (!d.queue) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullQueue);
    if (!d.swap_chain) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullSwapChain);
    if (d.queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)
        return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullQueue);

    DXGI_SWAP_CHAIN_DESC1 sd = {};
    HRESULT hr = d.swap_chain->GetDesc1(&sd);
    if (FAILED(hr)) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullSwapChain, hr);
    if (sd.SwapEffect != DXGI_SWAP_EFFECT_FLIP_DISCARD)
        return FgWinFail(FgStatus::Unsupported, FgWinRefusal::SwapChainNotFlipDiscard);
    if (!(sd.Flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT))
        return FgWinFail(FgStatus::Unsupported, FgWinRefusal::SwapChainNoWaitable);
    BOOL tearing = FALSE;
    {
        ComPtr<IDXGIFactory5> f5;
        if (SUCCEEDED(d.swap_chain->GetParent(IID_PPV_ARGS(&f5))) &&
            FAILED(f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
            tearing = FALSE;
    }
    if (d.pacing == FgDxgiPacing::Tearing) {
        if (!tearing) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NoTearing);
        if (!(sd.Flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING))
            return FgWinFail(FgStatus::Unsupported, FgWinRefusal::SwapChainNoTearingFlag);
    }

    FgEngineConfig cfg;
    cfg.probe = FgPresenterProbe{FgPresenterBackend::DxgiFlip, true, true, d.max_generated, FgStatus::Ok};
    cfg.slot_count = slots;
    cfg.refresh_period = d.refresh_period;
    std::unique_ptr<FgDxgiPresenter> p(new FgDxgiPresenter(d, cfg));
    p->info_.tearing_supported = tearing != FALSE;
    const FgWinResult r = p->Initialize();
    if (!r.Ok()) return r;
    *out = std::move(p);
    return FgWinOk();
}

FgWinResult FgDxgiPresenter::Initialize() {
    if (!timer_.Valid()) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::TimerUnavailable);
    copy_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    fence_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    drain_event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!copy_event_ || !fence_event_ || !drain_event_)
        return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation);
    latency_ = swap_->GetFrameLatencyWaitableObject();
    if (!latency_) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::SwapChainNoWaitable);
    HRESULT hr = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copy_fence_));
    if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation, hr);
    const FgWinResult r = AcquireChain();
    if (!r.Ok()) return r;
    if (desc_.pacing == FgDxgiPacing::Vsync) {
        if (!FgReadCompositorGrid(clock_map_, desc_.refresh_period, &info_.grid))
            return FgWinFail(FgStatus::Unsupported, FgWinRefusal::RefreshUnknown);
        grid_updated_at_ = FgInterruptNow();
    }
    if (!worker_.Start(this, L"wg-fg-dxgi-presenter")) return FgWinFail(FgStatus::Refused, FgWinRefusal::WorkerStart);
    return FgWinOk();
}

FgWinResult FgDxgiPresenter::AcquireChain() {
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    HRESULT hr = swap_->GetDesc1(&sd);
    if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::NullSwapChain, hr);
    back_buffers_.clear();
    back_buffers_.resize(sd.BufferCount);
    for (UINT i = 0; i < sd.BufferCount; ++i) {
        hr = swap_->GetBuffer(i, IID_PPV_ARGS(&back_buffers_[i]));
        if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation, hr);
    }
    const bool changed = sd.Width != info_.width || sd.Height != info_.height || sd.Format != info_.format;
    info_.buffer_count = sd.BufferCount;
    info_.width = sd.Width;
    info_.height = sd.Height;
    info_.format = sd.Format;
    if (slots_.empty() || changed) {
        const FgWinResult r = CreateSlots();
        if (!r.Ok()) return r;
    }
    const std::size_t ring = static_cast<std::size_t>(sd.BufferCount) + 1;
    if (allocators_.size() != ring) {
        allocators_.clear();
        allocators_.resize(ring);
        allocator_value_.assign(ring, 0);
        for (auto& a : allocators_) {
            hr = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a));
            if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation, hr);
        }
        list_.Reset();
        hr = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators_[0].Get(), nullptr,
                                        IID_PPV_ARGS(&list_));
        if (FAILED(hr) || FAILED(hr = list_->Close()))
            return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation, hr);
    }
    return FgWinOk();
}

FgWinResult FgDxgiPresenter::CreateSlots() {
    slots_.clear();
    slots_.resize(info_.slot_count);
    slot_copy_value_.assign(info_.slot_count, 0);
    for (auto& s : slots_) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        hp.CreationNodeMask = hp.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = info_.width;
        rd.Height = info_.height;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.Format = info_.format;
        rd.SampleDesc.Count = 1;
        rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                   (desc_.allow_unordered_access ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);
        const HRESULT hr = device_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                            nullptr, IID_PPV_ARGS(&s));
        if (FAILED(hr)) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::ResourceCreation, hr);
    }
    return FgWinOk();
}

FgDxgiPresenter::~FgDxgiPresenter() {
    worker_.Stop();
    std::lock_guard<std::mutex> lock(mutex_);
    // nothing may be destroyed while the GPU still copies from a slot into a back buffer or a queued present still
    // uses one: drain the queue past the last copy and present
    DrainQueueLocked(1000);
    if (claimed_) FgChainRegistry::Process().Release(chain_, this);
    claimed_ = false;
    back_buffers_.clear();
    if (latency_) CloseHandle(latency_);
    if (copy_event_) CloseHandle(copy_event_);
    if (fence_event_) CloseHandle(fence_event_);
    if (drain_event_) CloseHandle(drain_event_);
}

std::uint32_t FgDxgiPresenter::RegisterFence(ID3D12Fence* fence, FgWinResult* why) {
    if (why) *why = FgWinOk();
    if (!fence) {
        if (why) *why = FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::NullDevice);
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const std::uint32_t id = fences_.Add(fence);
    if (id == 0 && why) *why = FgWinFail(FgStatus::NoSlot, FgWinRefusal::ResourceCreation);
    return id;
}

ID3D12Resource* FgDxgiPresenter::SlotResource(std::uint32_t slot) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return slot < slots_.size() ? slots_[slot].Get() : nullptr;
}

FgStatus FgDxgiPresenter::ReserveRealSlot(std::uint32_t* slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.ReserveRealOnlySlot(slot);
}

void FgDxgiPresenter::ReleaseRealSlot() {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.ReleaseRealOnlySlot();
}

FgPresenterProbe FgDxgiPresenter::Probe() const {
    return FgPresenterProbe{FgPresenterBackend::DxgiFlip, true, true, desc_.max_generated, FgStatus::Ok};
}

FgDxgiPresenterInfo FgDxgiPresenter::Info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return info_;
}

std::uint32_t FgDxgiPresenter::FreeSlots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.FreeSlots();
}

bool FgDxgiPresenter::HoldsSwapChainBuffers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !back_buffers_.empty();
}

FgPresenterState FgDxgiPresenter::State() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.State();
}

FgPresentStats FgDxgiPresenter::Stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.Stats();
}

std::uint64_t FgDxgiPresenter::FirstQueuedPresentId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return engine_.FirstQueuedPresentId();
}

FgStatus FgDxgiPresenter::Attach(std::uint64_t chain_id, std::uint64_t epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (claimed_) return engine_.Refuse(FgStatus::WrongState);
    if (back_buffers_.empty()) {
        const FgWinResult r = AcquireChain();
        if (!r.Ok() || back_buffers_.empty()) return engine_.Refuse(FgStatus::WrongState);
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

FgStatus FgDxgiPresenter::BeginInterval(const FgIntervalPlan& plan, FgIntervalReservation* out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (desc_.pacing == FgDxgiPacing::Vsync && engine_.State() == FgPresenterState::Attached &&
        !FgPlanFitsRefresh(plan, info_.grid.period)) {
        if (out) *out = FgIntervalReservation();
        return engine_.Refuse(FgStatus::Unsupported);  // fixed refresh cannot show this cadence evenly
    }
    return engine_.BeginInterval(plan, FgInterruptNow(), out);
}

FgStatus FgDxgiPresenter::SubmitInterval(const FgIntervalSubmission& submission) {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        s = engine_.SubmitInterval(submission, FgInterruptNow());
    }
    worker_.Wake();
    return s;
}

FgStatus FgDxgiPresenter::PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real) {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!engine_.HasRealOnlyReservation()) return engine_.Refuse(FgStatus::WrongState);  // ReserveRealSlot first
        s = engine_.PresentRealOnly(plan, real);
    }
    worker_.Wake();
    return s;
}

FgStatus FgDxgiPresenter::RevokeGenerated(std::uint64_t interval_id, FgDropReason reason) {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::uint64_t from = 0;
        s = engine_.RevokeGenerated(interval_id, reason, FgInterruptNow(), &from);
        CancelLocalLocked(from);
    }
    worker_.Wake();
    return s;
}

std::uint32_t FgDxgiPresenter::CancelPresentsFrom(std::uint64_t present_id) {
    std::uint32_t n;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::uint64_t from = 0;
        n = engine_.CancelPresentsFrom(present_id, FgInterruptNow(), &from);
        CancelLocalLocked(from);
    }
    worker_.Wake();
    return n;
}

FgStatus FgDxgiPresenter::Quiesce() {
    FgStatus s;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::uint64_t from = 0;
        s = engine_.Quiesce(FgInterruptNow(), &from);
        CancelLocalLocked(from);
    }
    worker_.Wake();
    return s;
}

FgStatus FgDxgiPresenter::CompleteResize(bool resize_ok, std::uint64_t new_epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!CopiesIdleLocked()) return engine_.Refuse(FgStatus::Busy);
    const FgStatus s = engine_.CompleteResize(resize_ok, new_epoch);
    if (s != FgStatus::Ok) return s;
    const FgWinResult r = AcquireChain();  // the resized chain's buffers (and new slots when size / format changed)
    if (!r.Ok()) {
        back_buffers_.clear();
        engine_.MarkFailed();
        return engine_.Refuse(FgStatus::ResizeFailed);
    }
    have_vblank_ = false;
    local_.clear();
    return s;
}

FgStatus FgDxgiPresenter::OnDeviceLost() {
    std::lock_guard<std::mutex> lock(mutex_);
    const FgStatus s = engine_.OnDeviceLost();
    local_.clear();
    return s;
}

FgStatus FgDxgiPresenter::Detach() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint64_t from = 0;
    if (engine_.State() != FgPresenterState::Lost && engine_.State() != FgPresenterState::Detached &&
        !CopiesIdleLocked()) {
        if (engine_.State() == FgPresenterState::Attached) {
            engine_.Quiesce(FgInterruptNow(), &from);
            CancelLocalLocked(from);
        }
        worker_.Wake();
        return engine_.Refuse(FgStatus::Busy);
    }
    const FgStatus s = engine_.Detach(FgInterruptNow(), &from);
    CancelLocalLocked(from);
    if (s == FgStatus::Ok) {
        // the presents already on the game's queue finish before the chain is let go (the game may release the swap
        // chain right after Detach)
        DrainQueueLocked(1000);
        if (claimed_) FgChainRegistry::Process().Release(chain_, this);
        claimed_ = false;
        local_.clear();
    } else {
        worker_.Wake();
    }
    return s;
}

bool FgDxgiPresenter::WaitIdle(DWORD timeout_ms) {
    const ULONGLONG until = GetTickCount64() + timeout_ms;
    for (;;) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const FgPresenterState st = engine_.State();
            const bool drained = engine_.Drained() || st == FgPresenterState::Lost;
            const bool chain_free = st == FgPresenterState::Attached || back_buffers_.empty();
            if (drained && CopiesIdleLocked() && chain_free) return true;
        }
        if (GetTickCount64() >= until) return false;
        worker_.Wake();
        Sleep(1);
    }
}

// --- worker (all under mutex_ unless noted) ------------------------------------------------------------------------

bool FgDxgiPresenter::CopiesIdleLocked() const {
    return !copy_fence_ || copy_fence_->GetCompletedValue() >= copy_value_;
}

bool FgDxgiPresenter::DrainQueueLocked(DWORD timeout_ms) {
    if (!copy_fence_ || !queue_ || !drain_event_) return true;
    if (device_ && device_->GetDeviceRemovedReason() != S_OK) return false;  // nothing completes any more
    const std::uint64_t v = ++copy_value_;
    if (FAILED(queue_->Signal(copy_fence_.Get(), v))) return false;
    if (copy_fence_->GetCompletedValue() >= v) return true;
    if (FAILED(copy_fence_->SetEventOnCompletion(v, drain_event_))) return false;
    WaitForSingleObject(drain_event_, timeout_ms);
    return copy_fence_->GetCompletedValue() >= v;
}

void FgDxgiPresenter::LoseLocked() {
    engine_.OnDeviceLost();
    local_.clear();
}

void FgDxgiPresenter::CancelLocalLocked(std::uint64_t engine_from) {
    if (engine_from != 0) {
        std::deque<FgEnginePresent> keep;
        for (const FgEnginePresent& p : local_)
            if (p.present_id < engine_from) keep.push_back(p);
        local_.swap(keep);
    }
    ReleaseCopiedSlotsLocked();
}

void FgDxgiPresenter::ReleaseCopiedSlotsLocked() {
    const std::uint64_t done = copy_fence_ ? copy_fence_->GetCompletedValue() : 0;
    for (std::uint32_t slot = 0; slot < slot_copy_value_.size(); ++slot) {
        const FgEngineSlotState st = engine_.SlotState(slot);
        if (slot_copy_value_[slot] != 0) {
            // read by a copy: free once that copy finished (whether it is still on screen does not matter, the
            // back buffer holds its own copy)
            if (done >= slot_copy_value_[slot] && engine_.OnSlotReleased(slot, true)) slot_copy_value_[slot] = 0;
        } else if (st == FgEngineSlotState::Retired) {
            engine_.OnSlotReleased(slot, false);  // cancelled before any copy read it
        }
    }
}

void FgDxgiPresenter::StepLocked(FgTicks now) {
    const FgPresenterState st = engine_.State();
    if (st == FgPresenterState::Lost || st == FgPresenterState::Detached) return;
    if (device_->GetDeviceRemovedReason() != S_OK) {
        LoseLocked();
        return;
    }
    ReleaseCopiedSlotsLocked();
    if (desc_.pacing == FgDxgiPacing::Vsync && now - grid_updated_at_ > kGridRefresh) {
        grid_updated_at_ = now;
        FgRefreshGrid grid;
        DXGI_FRAME_STATISTICS fs = {};
        if (SUCCEEDED(swap_->GetFrameStatistics(&fs)) && fs.SyncQPCTime.QuadPart != 0) {
            grid.origin = clock_map_.FromQpc(fs.SyncQPCTime.QuadPart);  // this chain's own last vblank
            grid.period = info_.grid.period;
            info_.grid = grid;
        } else if (FgReadCompositorGrid(clock_map_, desc_.refresh_period, &grid)) {
            info_.grid = grid;
        }
    }
    engine_.Pump(now);
    std::vector<FgEnginePresent> out;
    engine_.TakeOutbox(&out);
    for (const FgEnginePresent& p : out) local_.push_back(p);
    // not attached any more: once the copies finished, the swap chain's buffers are let go so the game can resize
    if (engine_.State() != FgPresenterState::Attached && CopiesIdleLocked()) back_buffers_.clear();
}

bool FgDxgiPresenter::PlanHeadLocked(FgTicks now, Planned* out) {
    const bool vsync = desc_.pacing == FgDxgiPacing::Vsync;
    while (!local_.empty()) {
        const FgEnginePresent p = local_.front();
        if (engine_.SlotPresentId(p.slot) != p.present_id) {  // cancelled since
            local_.pop_front();
            continue;
        }
        Planned pl;
        pl.present = p;
        if (vsync) {
            FgTicks vblank = 0;
            FgTicks submit = 0;
            if (FgSnapToRefreshGrid(p.target, info_.grid, &vblank, &submit) != FgStatus::Ok) {
                engine_.MarkFailed();
                local_.clear();
                return false;
            }
            if (have_vblank_ && vblank <= last_vblank_) {
                if (!p.real) {  // two images on one vblank: the generated one yields
                    engine_.OnPresentDropped(p.present_id, FgDropReason::Pressure);
                    local_.pop_front();
                    continue;
                }
                vblank = last_vblank_ + info_.grid.period;
                submit = vblank - info_.grid.period / 2;
            }
            pl.issue_at = submit;
            pl.shown_at = vblank;
        } else {
            pl.issue_at = p.target;
            pl.shown_at = p.target;
        }
        if (!p.real) {
            bool late = vsync ? now > pl.shown_at : now > pl.shown_at + desc_.late_tolerance;
            if (!late && !vsync && local_.size() > 1) {
                const FgEnginePresent& next = local_[1];
                late = engine_.SlotPresentId(next.slot) == next.present_id && now >= next.target;
            }
            if (late) {  // it would be on screen for (almost) no time: drop it, the lease is free at once
                engine_.OnPresentDropped(p.present_id, FgDropReason::Late);
                local_.pop_front();
                continue;
            }
        }
        *out = pl;
        return true;
    }
    return false;
}

bool FgDxgiPresenter::RecordCopyAndPresentLocked(const Planned& pl) {
    const FgEnginePresent& p = pl.present;
    const UINT bbi = swap_->GetCurrentBackBufferIndex();
    if (bbi >= back_buffers_.size() || !back_buffers_[bbi] || p.slot >= slots_.size()) {
        engine_.MarkFailed();
        local_.clear();
        return false;
    }
    const std::size_t a = ring_ % allocators_.size();
    if (copy_fence_->GetCompletedValue() < allocator_value_[a]) {
        // the allocator's previous copy is still running; with the frame-latency wait this is rare and short. The
        // event is shared with the worker's copy wakeups, so the fence value itself decides.
        if (SUCCEEDED(copy_fence_->SetEventOnCompletion(allocator_value_[a], copy_event_)))
            WaitForSingleObject(copy_event_, 50);
        if (copy_fence_->GetCompletedValue() < allocator_value_[a]) return false;  // retried on the next wakeup
    }
    HRESULT hr = allocators_[a]->Reset();
    if (SUCCEEDED(hr)) hr = list_->Reset(allocators_[a].Get(), nullptr);
    if (FAILED(hr)) {
        if (FgIsDeviceLostHr(hr)) LoseLocked();
        else engine_.MarkFailed();
        return false;
    }
    ID3D12Resource* bb = back_buffers_[bbi].Get();
    ID3D12Resource* src = slots_[p.slot].Get();
    D3D12_RESOURCE_BARRIER b[2] = {};
    b[0].Type = b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b[0].Transition = {bb, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_PRESENT,
                       D3D12_RESOURCE_STATE_COPY_DEST};
    b[1].Transition = {src, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_COPY_SOURCE};
    list_->ResourceBarrier(2, b);
    list_->CopyResource(bb, src);
    for (D3D12_RESOURCE_BARRIER& x : b) {
        const D3D12_RESOURCE_STATES t = x.Transition.StateBefore;
        x.Transition.StateBefore = x.Transition.StateAfter;
        x.Transition.StateAfter = t;
    }
    list_->ResourceBarrier(2, b);
    hr = list_->Close();
    if (FAILED(hr)) {
        engine_.MarkFailed();
        return false;
    }
    // the GPU copy starts only after the producer finished the image
    ID3D12Fence* producer = fences_.Get(p.fence.fence);
    if (producer) queue_->Wait(producer, p.fence.value);
    ID3D12CommandList* lists[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    queue_->Signal(copy_fence_.Get(), ++copy_value_);
    allocator_value_[a] = copy_value_;
    slot_copy_value_[p.slot] = copy_value_;
    ++ring_;
    const bool tearing = desc_.pacing == FgDxgiPacing::Tearing;
    hr = swap_->Present(tearing ? 0 : 1, tearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
    if (tearing && hr == DXGI_ERROR_INVALID_CALL) {
        // DXGI_PRESENT_ALLOW_TEARING is invalid in exclusive fullscreen; retry without the tearing flag.
        hr = swap_->Present(0, 0);
    }
    if (FAILED(hr)) {
        if (FgIsDeviceLostHr(hr)) {
            LoseLocked();
        } else {
            engine_.MarkFailed();  // a failed present is a hard failure: refuse until Detach, no fallback
            local_.clear();
        }
        return false;
    }
    ++info_.presents;
    if (!tearing) {
        have_vblank_ = true;
        last_vblank_ = pl.shown_at;
    }
    // DXGI reports no per-present scan-out time here: tearing records the Present call time, VSync the vblank the
    // present was issued for
    engine_.OnPresentDisplayed(p.present_id, tearing ? FgInterruptNow() : pl.shown_at);
    return true;
}

// Called without the lock: waits for a frame-latency count (bounded, together with the stop event), then presents
// the head under the lock. A count that was taken but not used stays as a credit for the next present, so the
// latency object's count never drifts.
bool FgDxgiPresenter::PresentHead(HANDLE stop) {
    bool have_credit = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        have_credit = latency_credit_;
    }
    if (!have_credit) {
        HANDLE hs[2] = {stop, latency_};
        const DWORD w = WaitForMultipleObjects(2, hs, FALSE, kLatencyWaitMs);
        if (w == WAIT_OBJECT_0) return false;
        if (w != WAIT_OBJECT_0 + 1) {
            std::lock_guard<std::mutex> lock(mutex_);
            ++info_.latency_timeouts;
            Planned head;
            if (engine_.State() == FgPresenterState::Attached && PlanHeadLocked(FgInterruptNow(), &head) &&
                !head.present.real) {
                engine_.OnPresentDropped(head.present.present_id, FgDropReason::Pressure);  // no room in time
                local_.pop_front();
            }
            return true;  // a real image waits and retries
        }
        std::lock_guard<std::mutex> lock(mutex_);
        latency_credit_ = true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!latency_credit_) return true;
    const FgTicks now = FgInterruptNow();
    Planned head;
    if (engine_.State() != FgPresenterState::Attached || !PlanHeadLocked(now, &head)) return true;
    if (now + desc_.timer_lead < head.issue_at) return true;  // a new head with a later time: keep the credit
    if (!RecordCopyAndPresentLocked(head)) return true;
    latency_credit_ = false;
    local_.pop_front();
    return true;
}

void FgDxgiPresenter::Run(HANDLE stop, HANDLE wake) {
    constexpr FgTicks kSpinWindow = 5000;  // 0.5 ms: bridge waitable-timer wake jitter without burning CPU
    for (;;) {
        bool present_now = false;
        FgTicks spin_until = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const FgTicks now = FgInterruptNow();
            StepLocked(now);
            Planned head;
            if (engine_.State() == FgPresenterState::Attached && PlanHeadLocked(now, &head)) {
                const FgTicks target_at = head.issue_at - desc_.timer_lead;
                if (now >= target_at) {
                    present_now = true;
                } else if (target_at - now <= kSpinWindow) {
                    timer_.Cancel();
                    spin_until = target_at;
                } else {
                    timer_.ArmAt(target_at - kSpinWindow);
                }
            } else {
                const FgTicks deadline = engine_.NextDeadline();
                if (deadline != 0) timer_.ArmAt(deadline + 1);
                else timer_.Cancel();
            }
            FgReadyFence f;
            if (engine_.HeadPendingFence(&f)) {
                ID3D12Fence* fence = fences_.Get(f.fence);
                if (fence) fence->SetEventOnCompletion(f.value, fence_event_);
            }
            if (!CopiesIdleLocked()) copy_fence_->SetEventOnCompletion(copy_value_, copy_event_);
        }
        if (spin_until != 0) {
            while (FgInterruptNow() < spin_until) {
                if (WaitForSingleObject(stop, 0) == WAIT_OBJECT_0) {
                    timer_.Cancel();
                    return;
                }
                YieldProcessor();
            }
            present_now = true;
        }
        if (present_now) {
            if (!PresentHead(stop)) break;
            continue;
        }
        HANDLE hs[5] = {stop, wake, timer_.Handle(), fence_event_, copy_event_};
        if (WaitForMultipleObjects(5, hs, FALSE, 100) == WAIT_OBJECT_0) break;
    }
    timer_.Cancel();
}

}  // namespace wgfg
