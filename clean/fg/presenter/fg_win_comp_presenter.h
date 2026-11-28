// SPDX-License-Identifier: GPL-3.0-only
#pragma once
// Windows composition presenter: IFgPresenter over the presentation manager (presentation.h) with one target time
// per image, for the hardware flip queue.
//
//  - Presentation device: a small Direct3D 11 device on the game device's adapter (same LUID). Displayable surfaces
//    are a Direct3D 11 feature; the game's Direct3D 12 device produces every image.
//  - Zero copy: every display slot is ONE allocation. Direct3D 12 creates it (D3D12_HEAP_FLAG_SHARED |
//    D3D12_HEAP_FLAG_ALLOW_DISPLAY, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS), Direct3D 11 opens the same
//    allocation through an NT handle (it must come back D3D11_RESOURCE_MISC_SHARED_DISPLAYABLE) and registers it
//    with IPresentationManager::AddBufferFromResource. Nothing is copied.
//  - Synchronisation both ways:
//      D3D12 -> D3D11: the executor's producer fences are shared fences (D3D12_FENCE_FLAG_SHARED), opened on the
//      presentation device. A slot is presented only after its fence value completed on the CPU, and the D3D11
//      context also waits on the fence on the GPU before the present (ID3D11DeviceContext4::Wait), so an unfinished
//      image can never reach the display.
//      display -> D3D12: a slot is handed out again only after the display path is done with it: a terminal present
//      statistic (displayed and superseded, skipped or cancelled) AND IPresentationBuffer::IsAvailable. The
//      executor therefore never writes a buffer the compositor or the display still owns.
//  - Targets: SetTargetTime in interrupt time (QueryInterruptTimePrecise). On a fixed-refresh display every target
//    is snapped to the middle of its vblank's acceptance window (fg_present_engine.h FgSnapToRefreshGrid), so a
//    target phase near a vblank boundary cannot put two images into one refresh. With variable refresh the
//    scheduler's targets are used as they are. Composition never tears in either mode.
//  - ForceVSyncInterrupt(true) by default: Microsoft's composition swapchain guidance documents that disabling VSync
//    interrupts on hardware flip queue defers CPU feedback (present statistics, buffer availability events, retiring
//    fence). Keeping VSync interrupts on ensures timely statistics and slot recycling.
//  - Refuses (Unsupported, with a reason) when IsPresentationSupportedWithIndependentFlip or
//    D3D11_FEATURE_DISPLAYABLE is false, when the adapter reports no hardware flip queue, or when a slot does not open
//    displayable. It never falls back to a copy, a non-displayable path or another back end.
//  - A worker thread owns every display-path call: it waits on the producer fences, the statistics, buffer
//    availability, the lost event and a high-resolution timer for deadline drops, real-only pacing and occlusion
//    watchdogs. The game thread only reserves, submits and cancels (all non-blocking).
#include <d3d11_4.h>
#include <dcomp.h>
#include <dxgi1_6.h>
#include <presentation.h>

#include <memory>
#include <mutex>
#include <vector>

#include "fg_comp_backend.h"
#include "fg_display_cadence.h"
#include "fg_present_engine.h"
#include "fg_win_common.h"

namespace wgfg {

struct FgCompPresenterDesc {
    FgWorkerRegistration worker_registration;
    FgTicks min_interval = 0;
    ID3D12Device* device = nullptr;  // the game's device: it renders every image into the slots
    HWND hwnd = nullptr;             // the game window the composition visual is bound to (this process)
    // false: headless validation only. The surface is created but bound to no window, so nothing is shown; hwnd must
    // be null. Used by the offscreen GPU test.
    bool bind_window = true;
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
    bool hdr = false;               // HDR10 PQ when format is R10G10B10A2_UNORM
    bool variable_refresh = false;  // the output runs variable refresh: targets are used unsnapped
    FgTicks refresh_period = 0;     // fixed refresh: the scan-out period (0: measured from the compositor)
    std::uint32_t max_generated = kFgMaxGenerated;  // 1..5
    std::uint32_t slot_count = 0;   // 0: max_generated + 4; below max_generated + 3 is refused
    std::uint32_t max_pending_intervals = 2;
    bool force_vsync_interrupt = true;
    bool require_hw_flip_queue = true;  // refuse unless D3DKMT reports HwFlipQueueEnabled for the adapter
};

struct FgCompPresenterInfo {
    FgAdapterFlipCaps adapter;
    bool presentation_supported = false;
    bool independent_flip_supported = false;
    bool d3d11_displayable = false;
    std::uint32_t slot_count = 0;
    std::uint32_t d3d11_misc_flags = 0;  // misc flags of the opened slot textures (SHARED_DISPLAYABLE expected)
    bool simultaneous_access = false;
    FgRefreshGrid grid;
    std::uint64_t indep_flip_frames = 0;  // PresentStatisticsKind_IndependentFlipFrame count
    std::uint64_t comp_frames = 0;        // PresentStatisticsKind_CompositionFrame count
    std::uint64_t status_queued = 0;      // PresentStatus_Queued count
    std::uint64_t status_skipped = 0;     // PresentStatus_Skipped count
    std::uint64_t status_canceled = 0;    // PresentStatus_Canceled count
    std::uint64_t watchdog_retired = 0;   // queued presents retired via the 250 ms occlusion watchdog
    std::uint64_t visible_submitted = 0;  // presents submitted while the bound window was visible
    FgTicks first_visible_submit = 0;     // timestamp of the first visible-window present submission
    FgTicks last_visible_submit = 0;      // timestamp of the latest visible-window present submission
    bool force_vsync_interrupt = false;
    bool window_visible = false;
    bool window_cloaked = false;
    bool hwnd_is_root = true;
    bool primed = false;
    long last_present_hr = 0;
    std::uint64_t status_frame_nonzero = 0;            // PresentStatus reports with non-zero CompositionFrameId
    std::uint64_t status_queued_with_cfid = 0;         // PresentStatus_Queued with non-zero CompositionFrameId
    std::uint64_t status_first_cfid = 0;               // first non-zero CompositionFrameId on PresentStatus
    std::uint32_t status_target_count_max = 0;         // max DComp target count seen on PresentStatus / DComp probes
    std::uint64_t status_target_presented_nonzero = 0; // frames with DComp presentedStats.time != 0
    std::uint64_t status_target_completed_nonzero = 0; // frames with DComp completedStats.time != 0
    std::uint64_t retiring_fence_value = 0;            // IPresentationManager retiring fence completed value
    std::uint64_t display_time_iflip = 0;              // displayed times sourced from IndependentFlipFrame
    std::uint64_t display_time_comp = 0;               // displayed times sourced from CompositionFrame
    std::uint64_t display_time_dcomp_stats = 0;        // displayed times sourced from DComp target stats
    std::uint64_t display_time_estimated = 0;          // displayed times with no measured hardware/DComp timestamp
    std::uint64_t composed_drop_late = 0;              // generated images dropped pre-submit on composed grid (late)
    std::uint64_t composed_drop_collision = 0;         // generated images dropped pre-submit to avoid grid collision
    std::uint64_t grid_phase_bumps = 0;                // closed-loop vblank bumps from DWM PresentStatus_Queued
};

class FgWinCompPresenter final : public IFgPresenter, private IFgWorkerBody {
public:
    // Creates the presentation device, the slot buffers and the display path and starts the worker. On failure *out
    // stays null and the result names the refusal.
    static FgWinResult Create(const FgCompPresenterDesc& desc, std::unique_ptr<FgWinCompPresenter>* out);
    // Creates a composition presenter over an injected IFgCompBackend (for CPU tests with a fake compositor).
    static FgWinResult CreateWithBackend(const FgCompPresenterDesc& desc, std::unique_ptr<IFgCompBackend> backend,
                                         bool start_worker, std::unique_ptr<FgWinCompPresenter>* out);
    ~FgWinCompPresenter() override;
    FgWinCompPresenter(const FgWinCompPresenter&) = delete;
    FgWinCompPresenter& operator=(const FgWinCompPresenter&) = delete;

    // Advances the presenter once synchronously (used by single-threaded CPU tests with start_worker = false).
    void StepForTest();

    // Shares an executor fence (created with D3D12_FENCE_FLAG_SHARED) with the presentation device. Returns the id
    // for FgReadyFence::fence, 0 when refused (*why says why).
    std::uint32_t RegisterFence(ID3D12Fence* fence, FgWinResult* why);
    // The slot's Direct3D 12 resource. Valid only for slots reserved by BeginInterval / for the image the caller
    // presents; the executor writes it and signals its producer fence.
    ID3D12Resource* SlotResource(std::uint32_t slot) const;
    // A real-only present needs a slot too: reserves one (NoSlot when none is free). The game renders into it and
    // passes it in FgRealImage via PresentRealOnlySlot.
    FgStatus ReserveRealSlot(std::uint32_t* slot);
    void ReleaseRealSlot();
    // Size of the buffers created by the next successful CompleteResize.
    FgStatus SetResizeTarget(std::uint32_t width, std::uint32_t height) override;
    // Stale suffix (engine present ids, see FirstQueuedPresentId).
    std::uint32_t CancelPresentsFrom(std::uint64_t present_id);
    std::uint64_t FirstQueuedPresentId() const;
    // Waits (bounded) until no old work is left: before a resize or Detach. true when drained.
    bool WaitIdle(DWORD timeout_ms);
    FgCompPresenterInfo Info() const;
    FgDisplaySummary ReadDisplayCadence(bool window);
    std::uint32_t FreeSlots() const;
    DWORD WorkerThreadId() const { return worker_.ThreadId(); }
    void SetRealRetireCallback(FgRealRetireFn fn, void* ctx);
    void SetTraceCallback(FgPresentTraceFn fn, void* ctx);

    FgPresenterProbe Probe() const override;
    FgStatus Attach(std::uint64_t chain_id, std::uint64_t epoch) override;
    FgStatus BeginInterval(const FgIntervalPlan& plan, FgIntervalReservation* out) override;
    FgStatus SubmitInterval(const FgIntervalSubmission& submission) override;
    FgStatus PresentRealOnly(const FgIntervalPlan& plan, const FgRealImage& real) override;
    FgStatus RevokeGenerated(std::uint64_t interval_id, FgDropReason reason) override;
    FgStatus Quiesce() override;
    FgStatus CompleteResize(bool resize_ok, std::uint64_t new_epoch) override;
    FgStatus OnDeviceLost() override;
    FgStatus Detach() override;
    FgPresenterState State() const override;
    FgPresentStats Stats() const override;

private:
    struct Submitted {
        std::uint64_t engine_id = 0;
        std::uint64_t platform_id = 0;
        FgTicks vblank = 0;
        FgDisplayPresent display{};
        std::uint64_t comp_frame_id = 0;
        bool queued_confirmed = false;
        bool watchdog_cancelled = false;
    };

    FgWinCompPresenter(const FgCompPresenterDesc& desc, const FgEngineConfig& engine,
                       std::unique_ptr<IFgCompBackend> backend, bool start_worker);
    FgWinResult Initialize();
    FgTicks NowLocked() const;
    FgTicks RealOnlyLookahead() const;
    FgTicks PresentLookaheadLocked(bool real_only_plan) const;
    FgTicks RefreshPeriodLocked() const;
    void Run(HANDLE stop, HANDLE wake) override;
    void StepLocked();
    void DrainStatisticsLocked();
    void ResolveQueuedDisplaysLocked();
    void ReleaseSlotsLocked();
    void SubmitOutboxLocked();
    void CancelOnDisplayLocked(std::uint64_t engine_from);
    void LoseLocked();
    void UpdateGridLocked();
    void RefreshCompGridFromTimingLocked(const FgCompFrameTiming& timing);
    bool TakePlatformPresent(std::uint64_t platform_id, Submitted* out);
    void ForgetSubmittedLocked();

    FgCompPresenterDesc desc_;
    FgCompPresenterInfo info_;
    std::unique_ptr<IFgCompBackend> backend_;
    bool start_worker_ = true;
    FgHighResTimer timer_;

    mutable std::mutex mutex_;
    FgPresentEngine engine_;
    FgDisplayCadence display_;
    std::vector<FgEnginePresent> deferred_;
    std::vector<Submitted> submitted_;
    std::vector<std::uint32_t> scratch_slots_;
    bool have_vblank_ = false;
    FgTicks last_vblank_ = 0;
    FgTicks qlat_low_ = 0;
    FgTicks qlat_med_ = 0;
    FgTicks slip_med_ = 0;
    std::uint64_t last_slip_id_ = 0;
    FgTicks next_pace_wake_ = 0;
    FgTicks grid_updated_at_ = 0;
    std::uint64_t chain_ = 0;
    bool claimed_ = false;
    UINT resize_width_ = 0;
    UINT resize_height_ = 0;
    std::uint64_t last_displayed_platform_ = 0;  // newest platform present id with a displayed statistic
    bool need_buffers_ = false;  // buffers were handed to the compositor (detach / resize): make new ones first
    FgWorkerThread worker_;
};

}  // namespace wgfg
