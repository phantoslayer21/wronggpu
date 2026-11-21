// SPDX-License-Identifier: MIT
// Shared Windows pieces of the presenter back ends: see fg_win_common.h.
#include "fg_win_common.h"

#include <winternl.h>
#include <d3dkmthk.h>
#include <dcomp.h>
#include <dxgi.h>
#include <realtimeapiset.h>

#include <cstddef>
#include <system_error>

namespace wgfg {

const char* FgWinRefusalName(FgWinRefusal r) {
    switch (r) {
        case FgWinRefusal::None: return "none";
        case FgWinRefusal::NullDevice: return "null-device";
        case FgWinRefusal::NullQueue: return "null-queue";
        case FgWinRefusal::NullSwapChain: return "null-swap-chain";
        case FgWinRefusal::InvalidWindow: return "invalid-window";
        case FgWinRefusal::ForeignWindow: return "foreign-window";
        case FgWinRefusal::BadSize: return "bad-size";
        case FgWinRefusal::BadFormat: return "bad-format";
        case FgWinRefusal::SlotCount: return "slot-count";
        case FgWinRefusal::BadCount: return "bad-count";
        case FgWinRefusal::AdapterNotFound: return "adapter-not-found";
        case FgWinRefusal::AdapterQueryFailed: return "adapter-query-failed";
        case FgWinRefusal::NoHardwareFlipQueue: return "no-hardware-flip-queue";
        case FgWinRefusal::D3D11DeviceFailed: return "d3d11-device-failed";
        case FgWinRefusal::NoPresentationManager: return "no-presentation-manager";
        case FgWinRefusal::NoIndependentFlip: return "no-independent-flip";
        case FgWinRefusal::NotDisplayable: return "not-displayable";
        case FgWinRefusal::NoTearing: return "no-tearing";
        case FgWinRefusal::SwapChainNotFlipDiscard: return "swap-chain-not-flip-discard";
        case FgWinRefusal::SwapChainNoWaitable: return "swap-chain-no-waitable";
        case FgWinRefusal::SwapChainNoTearingFlag: return "swap-chain-no-tearing-flag";
        case FgWinRefusal::RefreshUnknown: return "refresh-unknown";
        case FgWinRefusal::TimerUnavailable: return "timer-unavailable";
        case FgWinRefusal::ResourceCreation: return "resource-creation";
        case FgWinRefusal::SharedHandle: return "shared-handle";
        case FgWinRefusal::CompositionTarget: return "composition-target";
        case FgWinRefusal::StatisticsUnavailable: return "statistics-unavailable";
        case FgWinRefusal::WorkerStart: return "worker-start";
        case FgWinRefusal::UnknownChoice: return "unknown-choice";
        case FgWinRefusal::Probe: return "probe";
    }
    return "?";
}

// --- Clock -----------------------------------------------------------------------------------------------------------

FgTicks FgInterruptNow() {
    ULONGLONG t = 0;
    QueryInterruptTimePrecise(&t);
    return static_cast<FgTicks>(t);
}

FgClockMap::FgClockMap() {
    LARGE_INTEGER f = {};
    LARGE_INTEGER c = {};
    QueryPerformanceFrequency(&f);
    ULONGLONG it = 0;
    QueryInterruptTimePrecise(&it);
    QueryPerformanceCounter(&c);
    freq_ = f.QuadPart > 0 ? f.QuadPart : 1;
    qpc0_ = c.QuadPart;
    it0_ = static_cast<FgTicks>(it);
}

FgTicks FgClockMap::FromQpc(std::int64_t qpc) const {
    const std::int64_t d = qpc - qpc0_;
    // whole seconds and remainder separately: no overflow for any realistic distance
    const std::int64_t secs = d / freq_;
    const std::int64_t rem = d % freq_;
    return it0_ + secs * 10000000 + (rem * 10000000) / freq_;
}

bool FgReadCompositorGrid(const FgClockMap& map, FgTicks period_override, FgRefreshGrid* out) {
    if (!out) return false;
    COMPOSITION_FRAME_ID id = 0;
    if (FAILED(DCompositionGetFrameId(COMPOSITION_FRAME_ID_COMPLETED, &id))) return false;
    COMPOSITION_FRAME_STATS st = {};
    COMPOSITION_TARGET_ID targets[4] = {};
    UINT count = 0;
    if (FAILED(DCompositionGetStatistics(id, &st, 4, targets, &count)) || st.targetTime == 0) return false;
    LARGE_INTEGER f = {};
    QueryPerformanceFrequency(&f);
    if (f.QuadPart <= 0) return false;
    const FgTicks measured =
        static_cast<FgTicks>((st.framePeriod * 10000000ull) / static_cast<std::uint64_t>(f.QuadPart));
    const FgTicks period = period_override > 0 ? period_override : measured;
    if (period <= 0) return false;
    out->origin = map.FromQpc(static_cast<std::int64_t>(st.targetTime));
    out->period = period;
    return true;
}

// --- High-resolution waitable timer ------------------------------------------------------------------------------

FgHighResTimer::FgHighResTimer() {
    // Documented since Windows 10 version 1803. No fallback to a coarse timer: a coarse timer cannot pace 2x..6x.
    handle_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                     TIMER_MODIFY_STATE | SYNCHRONIZE);
}

FgHighResTimer::~FgHighResTimer() {
    if (handle_) {
        CancelWaitableTimer(handle_);
        CloseHandle(handle_);
    }
}

bool FgHighResTimer::ArmAt(FgTicks when) {
    if (!handle_) return false;
    const FgTicks due = FgTimerDue(when, FgInterruptNow(), 0);
    LARGE_INTEGER li = {};
    li.QuadPart = due < 0 ? due : -1;  // relative; -1 tick = at once
    return SetWaitableTimerEx(handle_, &li, 0, nullptr, nullptr, nullptr, 0) != FALSE;
}

void FgHighResTimer::Cancel() {
    if (handle_) CancelWaitableTimer(handle_);
}

// --- Worker thread -------------------------------------------------------------------------------------------------

FgWorkerThread::FgWorkerThread() {
    stop_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    wake_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
}

FgWorkerThread::~FgWorkerThread() {
    Stop();
    if (stop_) CloseHandle(stop_);
    if (wake_) CloseHandle(wake_);
}

void FgWorkerThread::Entry(FgWorkerThread* self) {
    if (self->name_) SetThreadDescription(GetCurrentThread(), self->name_);
    self->body_->Run(self->stop_, self->wake_);
    self->running_.store(false);
}

bool FgWorkerThread::Start(IFgWorkerBody* body, const wchar_t* name) {
    if (!body || !stop_ || !wake_ || thread_.joinable()) return false;
    ResetEvent(stop_);
    body_ = body;
    name_ = name;
    running_.store(true);
    try {
        thread_ = std::thread(&FgWorkerThread::Entry, this);
    } catch (const std::system_error&) {
        running_.store(false);
        return false;
    }
    return true;
}

void FgWorkerThread::Stop() {
    if (!thread_.joinable()) return;
    SetEvent(stop_);
    thread_.join();
    running_.store(false);
}

void FgWorkerThread::Wake() {
    if (wake_) SetEvent(wake_);
}

// --- Producer fences -----------------------------------------------------------------------------------------------

std::uint32_t FgD3D12FenceTable::Add(ID3D12Fence* fence) {
    if (!fence) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t i = 0; i < fences_.size(); ++i)
        if (fences_[i].Get() == fence) return static_cast<std::uint32_t>(i + 1);
    if (fences_.size() >= kMax) return 0;
    fences_.emplace_back(fence);
    return static_cast<std::uint32_t>(fences_.size());
}

ID3D12Fence* FgD3D12FenceTable::Get(std::uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return id >= 1 && id <= fences_.size() ? fences_[id - 1].Get() : nullptr;
}

bool FgD3D12FenceTable::FenceValid(std::uint32_t fence) const { return Get(fence) != nullptr; }

std::uint64_t FgD3D12FenceTable::FenceCompleted(std::uint32_t fence) const {
    ID3D12Fence* f = Get(fence);
    // UINT64_MAX after device removal: the back ends detect the loss from their own calls, not from this value
    return f ? f->GetCompletedValue() : 0;
}

std::uint32_t FgD3D12FenceTable::Count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<std::uint32_t>(fences_.size());
}

void FgD3D12FenceTable::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    fences_.clear();
}

// --- Adapter flip-queue capability -----------------------------------------------------------------------------------

namespace {

template <class T>
NTSTATUS QueryAdapter(D3DKMT_HANDLE adapter, KMTQUERYADAPTERINFOTYPE type, T* out) {
    D3DKMT_QUERYADAPTERINFO q = {};
    q.hAdapter = adapter;
    q.Type = type;
    q.pPrivateDriverData = out;
    q.PrivateDriverDataSize = sizeof(T);
    return D3DKMTQueryAdapterInfo(&q);
}

}  // namespace

FgWinResult FgQueryAdapterFlipCaps(LUID luid, FgAdapterFlipCaps* out) {
    if (!out) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::AdapterQueryFailed);
    *out = FgAdapterFlipCaps();
    D3DKMT_OPENADAPTERFROMLUID open = {};
    open.AdapterLuid = luid;
    NTSTATUS st = D3DKMTOpenAdapterFromLuid(&open);
    if (st != 0) {
        out->status = static_cast<long>(st);
        return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterNotFound, static_cast<long>(st));
    }
    D3DKMT_WDDM_2_9_CAPS c29 = {};
    if (QueryAdapter(open.hAdapter, KMTQAITYPE_WDDM_2_9_CAPS, &c29) == 0) {
        out->hw_sch_queried = true;
        out->hw_sch_enabled = c29.HwSchEnabled != 0;
    }
    D3DKMT_WDDM_3_0_CAPS c30 = {};
    st = QueryAdapter(open.hAdapter, KMTQAITYPE_WDDM_3_0_CAPS, &c30);
    out->status = static_cast<long>(st);
    if (st == 0) {
        out->queried = true;
        out->hw_flip_queue_support_state = c30.HwFlipQueueSupportState;
        out->hw_flip_queue_enabled = c30.HwFlipQueueEnabled != 0;
        out->displayable_supported = c30.DisplayableSupported != 0;
    }
    D3DKMT_CLOSEADAPTER close = {};
    close.hAdapter = open.hAdapter;
    D3DKMTCloseAdapter(&close);
    if (st != 0) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterQueryFailed, static_cast<long>(st));
    return FgWinOk();
}

FgWinResult FgEnumAdapterLuids(std::vector<LUID>* out) {
    if (!out) return FgWinFail(FgStatus::InvalidArgument, FgWinRefusal::AdapterQueryFailed);
    out->clear();
    D3DKMT_ENUMADAPTERS2 e = {};
    NTSTATUS st = D3DKMTEnumAdapters2(&e);
    if (st != 0) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterQueryFailed, static_cast<long>(st));
    std::vector<D3DKMT_ADAPTERINFO> info(e.NumAdapters);
    e.pAdapters = info.data();
    st = D3DKMTEnumAdapters2(&e);
    if (st != 0) return FgWinFail(FgStatus::Unsupported, FgWinRefusal::AdapterQueryFailed, static_cast<long>(st));
    for (ULONG i = 0; i < e.NumAdapters; ++i) {
        out->push_back(info[i].AdapterLuid);
        D3DKMT_CLOSEADAPTER close = {};
        close.hAdapter = info[i].hAdapter;
        D3DKMTCloseAdapter(&close);
    }
    return FgWinOk();
}

bool FgIsDeviceLostHr(HRESULT hr) {
    return hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_HUNG ||
           hr == DXGI_ERROR_DRIVER_INTERNAL_ERROR || hr == static_cast<HRESULT>(0x88810001L) /* PRESENTATION_ERROR_LOST */;
}

}  // namespace wgfg
