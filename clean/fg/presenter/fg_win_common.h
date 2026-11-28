// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Shared Windows pieces of the presenter back ends: explicit refusal reasons, the interrupt-time clock, the
// high-resolution waitable timer, the presenter worker thread, the D3D12 producer-fence table and the read-only
// adapter flip-queue capability query (D3DKMT).
//
// Written from Microsoft's public documentation of the Win32, DXGI, Direct3D 11/12, DirectComposition and
// presentation APIs. No third-party code.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <d3d12.h>
#include <d3d11_4.h>
#include <dcomp.h>
#include <dxgi1_6.h>
#include <presentation.h>
#include <wrl/client.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "fg_present_engine.h"

namespace wgfg {

// Why a back end or a selection refused. Every refusal names one; nothing falls back to another route on its own.
enum class FgWinRefusal : std::uint8_t {
    None = 0,
    NullDevice,               // no D3D12 device given
    NullQueue,                // no D3D12 command queue given
    NullSwapChain,            // no swap chain given
    InvalidWindow,            // the HWND is null or not a window
    ForeignWindow,            // the window belongs to another process
    BadSize,                  // zero / oversized width or height
    BadFormat,                // a format the display path cannot scan out
    SlotCount,                // slot pool below N + 3 or above 32
    BadCount,                 // max_generated outside 1..5
    AdapterNotFound,          // no DXGI adapter with the device's LUID
    AdapterQueryFailed,       // the D3DKMT capability query failed
    NoHardwareFlipQueue,      // D3DKMT_WDDM_3_0_CAPS.HwFlipQueueEnabled is 0
    D3D11DeviceFailed,        // the presentation device could not be created
    NoPresentationManager,    // IsPresentationSupported is false or the factory failed
    NoIndependentFlip,        // IsPresentationSupportedWithIndependentFlip is false
    NotDisplayable,           // D3D11_FEATURE_DISPLAYABLE is false or a buffer was not opened displayable
    NoTearing,                // DXGI_FEATURE_PRESENT_ALLOW_TEARING is false
    SwapChainNotFlipDiscard,  // the swap chain is not DXGI_SWAP_EFFECT_FLIP_DISCARD
    SwapChainNoWaitable,      // created without DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT
    SwapChainNoTearingFlag,   // created without DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING (tearing mode)
    RefreshUnknown,           // a fixed-refresh mode without a refresh period
    TimerUnavailable,         // no high-resolution waitable timer
    ResourceCreation,         // a slot texture, fence, command object or event could not be created
    SharedHandle,             // a shared handle could not be created or opened
    CompositionTarget,        // the DirectComposition target / visual / surface could not be set up
    StatisticsUnavailable,    // present statistics could not be enabled
    WorkerStart,              // the presenter worker thread did not start
    UnknownChoice,            // an unknown back-end choice
    Probe,                    // the capability probe had not run / failed for this route
};
const char* FgWinRefusalName(FgWinRefusal r);

struct FgWinResult {
    FgStatus status = FgStatus::Ok;
    FgWinRefusal refusal = FgWinRefusal::None;
    long hr = 0;  // the HRESULT / NTSTATUS behind the refusal, 0 when none
    bool Ok() const { return status == FgStatus::Ok; }
};
inline FgWinResult FgWinOk() { return FgWinResult{}; }
inline FgWinResult FgWinFail(FgStatus s, FgWinRefusal r, long hr = 0) {
    FgWinResult out;
    out.status = s;
    out.refusal = r;
    out.hr = hr;
    return out;
}

// --- Clock -----------------------------------------------------------------------------------------------------------

// Interrupt time in 100 ns ticks (QueryInterruptTimePrecise): the clock IPresentationManager::SetTargetTime takes.
FgTicks FgInterruptNow();

// Maps QPC values to interrupt-time ticks through one paired sample (both clocks advance at the same rate; QPC has
// its own origin and frequency).
class FgClockMap {
public:
    FgClockMap();
    FgTicks FromQpc(std::int64_t qpc) const;
    FgTicks DurationFromQpc(std::uint64_t qpc_ticks) const;

private:
    std::int64_t qpc0_ = 0;
    std::int64_t freq_ = 1;
    FgTicks it0_ = 0;
};

// The compositor's scan-out grid: the last completed composition frame's target time (a vblank) as origin and its
// frame period, unless `period_override` > 0 (the exact period from the display mode). false when the compositor
// reports no frame.
bool FgReadCompositorGrid(const FgClockMap& map, FgTicks period_override, FgRefreshGrid* out);

// DirectComposition and presentation entry points, resolved from dcomp.dll when first used (GetProcAddress by name, never a static
// import: dcomp.lib binds the undocumented statistics exports by ordinal, which a Windows without them (Server 2022)
// refuses to load the module for). Each returns HRESULT_FROM_WIN32(ERROR_PROC_NOT_FOUND) when dcomp.dll or the export
// is absent, so a caller treats it like any other failure.
HRESULT FgDCompGetFrameId(COMPOSITION_FRAME_ID_TYPE type, COMPOSITION_FRAME_ID* id);
HRESULT FgDCompGetStatistics(COMPOSITION_FRAME_ID id, COMPOSITION_FRAME_STATS* stats, UINT target_count,
                             COMPOSITION_TARGET_ID* targets, UINT* actual_count);
HRESULT FgDCompGetTargetStatistics(COMPOSITION_FRAME_ID id, const COMPOSITION_TARGET_ID* target,
                                   COMPOSITION_TARGET_STATS* stats);
HRESULT FgDCompCreateSurfaceHandle(DWORD access, SECURITY_ATTRIBUTES* attributes, HANDLE* handle);
HRESULT FgDCompCreatePresentationFactory(IUnknown* d3d_device, REFIID iid, void** factory);
HRESULT FgDCompCreateDevice3(IUnknown* rendering_device, REFIID iid, void** device);

// --- High-resolution waitable timer ------------------------------------------------------------------------------

class FgHighResTimer {
public:
    FgHighResTimer();
    ~FgHighResTimer();
    FgHighResTimer(const FgHighResTimer&) = delete;
    FgHighResTimer& operator=(const FgHighResTimer&) = delete;
    bool Valid() const { return handle_ != nullptr; }
    HANDLE Handle() const { return handle_; }
    // Arms the timer to fire at absolute interrupt time `when` (at once when it has passed). false on failure.
    bool ArmAt(FgTicks when);
    void Cancel();

private:
    HANDLE handle_ = nullptr;
};

// --- Worker thread -------------------------------------------------------------------------------------------------

// The body a worker runs. Run returns when `stop` is signalled (it must wait on it in every blocking wait).
class IFgWorkerBody {
public:
    virtual ~IFgWorkerBody() = default;
    virtual void Run(HANDLE stop, HANDLE wake) = 0;
};

// One presenter worker: a std::thread started on a named static entry point (no lambda, so nothing that can throw
// is copied into a closure under /Zc:lambda), a manual-reset stop event and an auto-reset wake event.
struct FgWorkerRegistration {
    void (*register_thread)(DWORD) = nullptr;
    void (*unregister_thread)(DWORD) = nullptr;
    void (*report_priority)(void*, DWORD, bool, bool) = nullptr;
    void* report_ctx = nullptr;
};

class FgWorkerThread {
public:
    FgWorkerThread();
    ~FgWorkerThread();  // stops and joins
    FgWorkerThread(const FgWorkerThread&) = delete;
    FgWorkerThread& operator=(const FgWorkerThread&) = delete;

    // false when already running, the events are missing or the thread could not start.
    bool Start(IFgWorkerBody* body, const wchar_t* name, FgWorkerRegistration registration = {});
    void Stop();  // idempotent; joins
    void Wake();
    bool Running() const { return running_.load(); }
    bool Joinable() const { return thread_.joinable(); }
    DWORD ThreadId() const { return thread_id_.load(); }

private:
    static void Entry(FgWorkerThread* self);

    HANDLE stop_ = nullptr;
    HANDLE wake_ = nullptr;
    IFgWorkerBody* body_ = nullptr;
    const wchar_t* name_ = nullptr;
    FgWorkerRegistration registration_;
    std::atomic<bool> running_{false};
    std::atomic<DWORD> thread_id_{0};
    std::thread thread_;
};

// --- Producer fences -----------------------------------------------------------------------------------------------

// The executor's D3D12 fences, by small id (FgReadyFence::fence). Thread-safe: ID3D12Fence::GetCompletedValue may
// be called from any thread.
class FgD3D12FenceTable final : public IFgFenceSource {
public:
    // 0 when the fence is null or the table is full.
    std::uint32_t Add(ID3D12Fence* fence);
    ID3D12Fence* Get(std::uint32_t id) const;
    bool FenceValid(std::uint32_t fence) const override;
    std::uint64_t FenceCompleted(std::uint32_t fence) const override;
    std::uint32_t Count() const;
    void Clear();

private:
    static constexpr std::uint32_t kMax = 64;
    mutable std::mutex mutex_;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Fence>> fences_;
};

// --- Adapter flip-queue capability (read only) ---------------------------------------------------------------------

struct FgAdapterFlipCaps {
    bool queried = false;                 // the WDDM 3.0 query answered
    long status = 0;                      // NTSTATUS of the query (0 = success)
    std::uint32_t hw_flip_queue_support_state = 0;  // D3DKMT_WDDM_3_0_CAPS.HwFlipQueueSupportState (2 = stable)
    bool hw_flip_queue_enabled = false;   // D3DKMT_WDDM_3_0_CAPS.HwFlipQueueEnabled
    bool displayable_supported = false;   // D3DKMT_WDDM_3_0_CAPS.DisplayableSupported
    bool hw_sch_queried = false;
    bool hw_sch_enabled = false;          // D3DKMT_WDDM_2_9_CAPS.HwSchEnabled
};

// Opens the adapter by LUID through the documented D3DKMT interface and reads its WDDM 2.9 / 3.0 capability bits. No
// device is created and nothing is changed.
FgWinResult FgQueryAdapterFlipCaps(LUID luid, FgAdapterFlipCaps* out);
// Every adapter's LUID (D3DKMTEnumAdapters2), read only.
FgWinResult FgEnumAdapterLuids(std::vector<LUID>* out);

// true for the HRESULTs that mean the device or the presentation path is gone.
bool FgIsDeviceLostHr(HRESULT hr);

}  // namespace wgfg
