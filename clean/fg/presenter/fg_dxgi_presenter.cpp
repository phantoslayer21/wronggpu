// SPDX-License-Identifier: GPL-3.0-only
// DXGI flip-discard presenter: see fg_dxgi_presenter.h.
#include "fg_dxgi_presenter.h"

#include <cstddef>

namespace wgfg {

using Microsoft::WRL::ComPtr;

namespace {

constexpr std::uint32_t kMaxSlots = 32;
constexpr FgTicks kGridRefresh = 10000000;  // re-read the vblank phase once a second
constexpr DWORD kLatencyWaitMs = 10;

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
    if (d.refresh_period < 0 || d.min_interval < 0 || d.late_tolerance < 0 || d.timer_lead < 0)
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
    cfg.max_pending_intervals = d.max_pending_intervals >= 1 ? d.max_pending_intervals : 2u;
    cfg.refresh_period = d.refresh_period;
    if (d.pacing == FgDxgiPacing::Tearing)
        cfg.pending_late_tolerance = d.late_tolerance > 40000 ? d.late_tolerance : 40000;
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
    if (!worker_.Start(this, L"wg-fg-dxgi-presenter", desc_.worker_registration)) return FgWinFail(FgStatus::Refused, FgWinRefusal::WorkerStart);
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
    const std::size_t ring = (static_cast<std::size_t>(sd.BufferCount) + 2) < 8 ? 8u : static_cast<std::size_t>(sd.BufferCount) + 2;
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

void FgDxgiPresenter::SetRealRetireCallback(FgRealRetireFn fn, void* ctx) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.SetRealRetireCallback(fn, ctx);
}

void FgDxgiPresenter::SetTraceCallback(FgPresentTraceFn fn, void* ctx) {
    std::lock_guard<std::mutex> lock(mutex_);
    engine_.SetTraceCallback(fn, ctx);
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
    if (claimed_ || back_buffers_.empty()) return engine_.Refuse(FgStatus::WrongState);
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
    if (presenting_ || !CopiesIdleLocked()) return engine_.Refuse(FgStatus::Busy);
    const FgStatus s = engine_.CompleteResize(resize_ok, new_epoch);
    if (s != FgStatus::Ok) return s;
    const FgWinResult r = AcquireChain();  // the resized chain's buffers (and new slots when size / format changed)
    if (!r.Ok()) {
        back_buffers_.clear();
        engine_.MarkFailed();
        return engine_.Refuse(FgStatus::ResizeFailed);
    }
    have_vblank_ = false;
    quiesce_fence_signalled_ = false;
    latency_credit_ = false;
    last_present_start_ = 0;
    last_present_done_ = 0;
    local_.clear();
    return s;
}

FgStatus FgDxgiPresenter::OnDeviceLost() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (presenting_) {
        device_lost_pending_ = true;
        return FgStatus::Ok;
    }
    const FgStatus s = engine_.OnDeviceLost();
    local_.clear();
    return s;
}

FgStatus FgDxgiPresenter::Detach() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::uint64_t from = 0;
    if (engine_.State() != FgPresenterState::Lost && engine_.State() != FgPresenterState::Detached &&
        (presenting_ || !CopiesIdleLocked())) {
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
            if (!presenting_ && drained && CopiesIdleLocked() && chain_free) return true;
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
        if (st == FgEngineSlotState::Presenting) continue;
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
    if (st == FgPresenterState::Attached && desc_.pacing == FgDxgiPacing::Vsync &&
        now - grid_updated_at_ > kGridRefresh) {
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
    if (engine_.State() != FgPresenterState::Attached && !presenting_ && !back_buffers_.empty()) {
        // The copy fence was signalled before native Present. Resize also needs completion of work DXGI queued.
        if (!quiesce_fence_signalled_) {
            const HRESULT hr = queue_->Signal(copy_fence_.Get(), copy_value_ + 1);
            if (FAILED(hr)) {
                LoseLocked();
                return;
            }
            ++copy_value_;
            quiesce_fence_signalled_ = true;
        }
        if (CopiesIdleLocked()) back_buffers_.clear();
    }
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
            const bool next_ready =
                local_.size() > 1 && engine_.SlotPresentId(local_[1].slot) == local_[1].present_id;
            const FgTicks next_target = next_ready ? local_[1].target : engine_.HeadPendingTarget();
            const FgTicks earliest = FgTearingEarliestIssue(last_present_start_, last_present_done_, desc_.min_interval,
                                                            desc_.refresh_period);
            const FgTearingPlanDecision td = FgPlanTearingImage(p.target, p.real, now, earliest, desc_.min_interval,
                                                                desc_.late_tolerance, next_target > p.target, next_target);
            if (td.drop_late) {  // past tolerance or would push next past its slip allowance: drop and free lease
                engine_.OnPresentDropped(p.present_id, FgDropReason::Late);
                local_.pop_front();
                continue;
            }
            pl.issue_at = td.issue_at;
            pl.shown_at = p.target;
        }
        if (vsync && !p.real && now > pl.shown_at) {
            engine_.OnPresentDropped(p.present_id, FgDropReason::Late);
            local_.pop_front();
            continue;
        }
        *out = pl;
        return true;
    }
    return false;
}

// Called without the lock: waits for a frame-latency count (bounded, together with the stop event), snapshots the
// copy/present objects under mutex_, then runs the allocator wait, CopyResource, ExecuteCommandLists and
// swap_->Present() OUTSIDE mutex_ (N-S1: avoids blocking game-thread BeginInterval/SubmitInterval and avoids any
// lock-ordering conflict with OptiScaler's wrapper mutex during overlay rendering or resize).
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
            const FgTicks now = FgInterruptNow();
            Planned head;
            if (engine_.State() == FgPresenterState::Attached && PlanHeadLocked(now, &head) && head.present.real &&
                now - head.present.target >= 2500000) {
                ++info_.occlusion_retired_real;
                engine_.OnPresentSkipped(head.present.present_id);
                local_.pop_front();
                ReleaseCopiedSlotsLocked();
            }
            return true;  // replan against the current queue and generated deadlines
        }
        std::lock_guard<std::mutex> lock(mutex_);
        latency_credit_ = true;
    }
    Planned head;
    ComPtr<ID3D12Resource> bb;
    ComPtr<ID3D12Resource> src;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> copy_fence;
    ComPtr<ID3D12Fence> producer;
    ComPtr<IDXGISwapChain4> swap;
    std::size_t a = 0;
    std::uint64_t copy_val = 0;
    FgTicks min_present_at = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!latency_credit_) return true;
        const FgTicks now = FgInterruptNow();
        if (engine_.State() != FgPresenterState::Attached || !PlanHeadLocked(now, &head)) return true;
        // Keep the wake lead even when the spacing floor controls so copy recording can start early.
        if (now + desc_.timer_lead < head.issue_at) return true;  // a new head with a later time: keep the credit
        min_present_at = FgTearingEarliestIssue(last_present_start_, last_present_done_, desc_.min_interval,
                                                desc_.refresh_period);
        const UINT bbi = swap_->GetCurrentBackBufferIndex();
        if (bbi >= back_buffers_.size() || !back_buffers_[bbi] || head.present.slot >= slots_.size() ||
            allocators_.empty()) {
            engine_.MarkFailed();
            local_.clear();
            return true;
        }
        a = ring_ % allocators_.size();
        // Wait for the allocator before taking the lease. The run loop retries while the copy fence drains.
        if (copy_fence_->GetCompletedValue() < allocator_value_[a]) return true;
        if (!engine_.LeasePresent(head.present.slot, head.present.present_id)) return true;
        local_.pop_front();
        alloc = allocators_[a];
        bb = back_buffers_[bbi];
        src = slots_[head.present.slot];
        list = list_;
        queue = queue_;
        copy_fence = copy_fence_;
        producer = fences_.Get(head.present.fence.fence);
        swap = swap_;
        copy_val = ++copy_value_;
        ++ring_;
        presenting_ = true;
    }

    // ---- Unlocked GPU copy + Present ----
    HRESULT hr = alloc->Reset();
    if (SUCCEEDED(hr)) hr = list->Reset(alloc.Get(), nullptr);
    if (SUCCEEDED(hr)) {
        D3D12_RESOURCE_BARRIER b[2] = {};
        b[0].Type = b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[0].Transition = {bb.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_PRESENT,
                           D3D12_RESOURCE_STATE_COPY_DEST};
        b[1].Transition = {src.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(2, b);
        list->CopyResource(bb.Get(), src.Get());
        for (D3D12_RESOURCE_BARRIER& x : b) {
            const D3D12_RESOURCE_STATES t = x.Transition.StateBefore;
            x.Transition.StateBefore = x.Transition.StateAfter;
            x.Transition.StateAfter = t;
        }
        list->ResourceBarrier(2, b);
        hr = list->Close();
    }
    if (SUCCEEDED(hr) && producer) hr = queue->Wait(producer.Get(), head.present.fence.value);
    bool copied = false;
    if (SUCCEEDED(hr)) {
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        copied = true;
        hr = queue->Signal(copy_fence.Get(), copy_val);
    }
    const bool tearing = desc_.pacing == FgDxgiPacing::Tearing;
    if (SUCCEEDED(hr) && tearing && min_present_at != 0) {
        // The lease is taken at most timer_lead before issue. Copy preparation can overlap that wait.
        while (FgInterruptNow() < min_present_at) YieldProcessor();
    }
    FgTicks present_started = 0;
    if (SUCCEEDED(hr)) {
        present_started = FgInterruptNow();
        hr = swap->Present(tearing ? 0 : 1, tearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
        if (tearing && hr == DXGI_ERROR_INVALID_CALL) {
            // DXGI_PRESENT_ALLOW_TEARING is invalid in exclusive fullscreen; retry without the tearing flag.
            present_started = FgInterruptNow();
            hr = swap->Present(0, 0);
        }
    }
    const FgTicks present_done = FgInterruptNow();

    std::lock_guard<std::mutex> lock(mutex_);
    presenting_ = false;
    if (copied) {
        allocator_value_[a] = copy_val;
        slot_copy_value_[head.present.slot] = copy_val;
    } else {
        --copy_value_;
    }
    if (device_lost_pending_ || FgIsDeviceLostHr(hr)) {
        device_lost_pending_ = false;
        LoseLocked();
        return true;
    }
    const bool shown = SUCCEEDED(hr) && hr != DXGI_STATUS_OCCLUDED;
    // Keep the submission and display estimates anchored to the return from the Present call.
    engine_.FinishLeasedPresent(head.present.slot, SUCCEEDED(hr) ? present_done : 0,
                                tearing ? present_done : head.shown_at, shown);
    if (FAILED(hr)) {
        engine_.MarkFailed();
        local_.clear();
        return true;
    }
    ++info_.presents;
    const FgTicks duration = present_done - present_started;
    info_.present_call_total += duration;
    if (duration > info_.present_call_max) info_.present_call_max = duration;
    last_present_start_ = present_started;
    last_present_done_ = present_done;
    if (!tearing) {
        have_vblank_ = true;
        last_vblank_ = head.shown_at;
    }
    latency_credit_ = false;
    return true;
}

void FgDxgiPresenter::Run(HANDLE stop, HANDLE wake) {
    for (;;) {
        bool present_now = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const FgTicks now = FgInterruptNow();
            StepLocked(now);
            Planned head;
            if (engine_.State() == FgPresenterState::Attached && PlanHeadLocked(now, &head)) {
                // head.issue_at already includes both spacing floors.
                const FgTicks target_at = head.issue_at > desc_.timer_lead ? head.issue_at - desc_.timer_lead : 0;
                const FgTicks wake_at =
                    desc_.pacing == FgDxgiPacing::Tearing && last_present_done_ > target_at ? last_present_done_ : target_at;
                const auto next_allocator = ring_ % allocators_.size();
                if (copy_fence_->GetCompletedValue() < allocator_value_[next_allocator]) {
                    timer_.Cancel();
                } else if (now >= wake_at) {
                    present_now = true;
                } else {
                    timer_.ArmAt(wake_at);
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
            if (engine_.PendingConsumptionFence(&f)) {
                if (ID3D12Fence* fence = fences_.Get(f.fence)) fence->SetEventOnCompletion(f.value, fence_event_);
            }
            if (!CopiesIdleLocked()) copy_fence_->SetEventOnCompletion(copy_value_, copy_event_);
        }
        if (present_now) {
            if (WaitForSingleObject(stop, 0) == WAIT_OBJECT_0 || !PresentHead(stop)) break;
            continue;
        }
        HANDLE hs[] = {stop, wake, timer_.Handle(), fence_event_, copy_event_};
        if (WaitForMultipleObjects(5, hs, FALSE, 100) == WAIT_OBJECT_0) break;
    }
    timer_.Cancel();
}

}  // namespace wgfg
