// SPDX-License-Identifier: MIT
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
//  - ForceVSyncInterrupt(false) by default: the interrupt is CPU feedback only and the live record measured the
//    even windowed cadence with it off.
//  - Refuses (Unsupported, with a reason) when IsPresentationSupportedWithIndependentFlip or
//    D3D11_FEATURE_DISPLAYABLE is false, when the adapter reports no hardware flip queue, or when a slot does not open
//    displayable. It never falls back to a copy, a non-displayable path or another back end.
//  - A worker thread owns every display-path call: it waits on the producer fences, the statistics, buffer
//    availability, the lost event and a high-resolution timer for deadline drops. The game thread only reserves,
//    submits and cancels (all non-blocking).
#include <d3d11_4.h>
#include <dcomp.h>
#include <dxgi1_6.h>
#include <presentation.h>

#include <memory>
#include <mutex>
#include <vector>

#include "fg_present_engine.h"
#include "fg_win_common.h"

namespace wgfg {

struct FgCompPresenterDesc {
    ID3D12Device* device = nullptr;  // the game's device: it renders every image into the slots
    HWND hwnd = nullptr;             // the game window the composition visual is bound to (this process)
    // false: headless validation only. The surface is created but bound to no window, so nothing is shown; hwnd must
    // be null. Used by the offscreen GPU test.
    bool bind_window = true;
    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_B8G8R8A8_UNORM;
    bool variable_refresh = false;  // the output runs variable refresh: targets are used unsnapped
    FgTicks refresh_period = 0;     // fixed refresh: the scan-out period (0: measured from the compositor)
    std::uint32_t max_generated = kFgMaxGenerated;  // 1..5
    std::uint32_t slot_count = 0;   // 0: max_generated + 4; below max_generated + 3 is refused
    bool force_vsync_interrupt = false;
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
};

class FgWinCompPresenter final : public IFgPresenter, private IFgWorkerBody {
public:
    // Creates the presentation device, the slot buffers and the display path and starts the worker. On failure *out
    // stays null and the result names the refusal.
    static FgWinResult Create(const FgCompPresenterDesc& desc, std::unique_ptr<FgWinCompPresenter>* out);
    ~FgWinCompPresenter() override;
    FgWinCompPresenter(const FgWinCompPresenter&) = delete;
    FgWinCompPresenter& operator=(const FgWinCompPresenter&) = delete;

    // Shares an executor fence (created with D3D12_FENCE_FLAG_SHARED) with the presentation device. Returns the id
    // for FgReadyFence::fence, 0 when refused (*why says why).
    std::uint32_t RegisterFence(ID3D12Fence* fence, FgWinResult* why);
    // The slot's Direct3D 12 resource. Valid only for slots reserved by BeginInterval / for the image the caller
    // presents; the executor writes it and signals its producer fence.
    ID3D12Resource* SlotResource(std::uint32_t slot) const;
    // A real-only present needs a slot too: reserves one (NoSlot when none is free). The game renders into it and
    // passes it in FgRealImage via PresentRealOnlySlot.
    FgStatus ReserveRealSlot(std::uint32_t* slot);
    // Size of the buffers created by the next successful CompleteResize.
    FgStatus SetResizeTarget(UINT width, UINT height);
    // Stale suffix (engine present ids, see FirstQueuedPresentId).
    std::uint32_t CancelPresentsFrom(std::uint64_t present_id);
    std::uint64_t FirstQueuedPresentId() const;
    // Waits (bounded) until no old work is left: before a resize or Detach. true when drained.
    bool WaitIdle(DWORD timeout_ms);
    FgCompPresenterInfo Info() const;
    std::uint32_t FreeSlots() const;

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
    struct Buffer {
        Microsoft::WRL::ComPtr<ID3D12Resource> resource;
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        Microsoft::WRL::ComPtr<IPresentationBuffer> buffer;
        HANDLE available = nullptr;
    };
    struct Submitted {
        std::uint64_t engine_id;
        std::uint64_t platform_id;
        std::uint32_t slot;
    };

    explicit FgWinCompPresenter(const FgCompPresenterDesc& desc, const FgEngineConfig& engine);
    FgWinResult Initialize();
    FgWinResult CreateBuffers(UINT width, UINT height);
    void DestroyBuffers();
    void Run(HANDLE stop, HANDLE wake) override;
    void StepLocked();
    void DrainStatisticsLocked();
    void ReleaseSlotsLocked();
    void SubmitOutboxLocked();
    void CancelOnDisplayLocked(std::uint64_t engine_from);
    void LoseLocked();
    void UpdateGridLocked();
    bool EngineIdForPlatform(std::uint64_t platform_id, std::uint64_t* engine_id, bool erase);

    FgCompPresenterDesc desc_;
    FgCompPresenterInfo info_;
    Microsoft::WRL::ComPtr<ID3D12Device> device12_;
    Microsoft::WRL::ComPtr<ID3D11Device5> device11_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext4> context11_;
    Microsoft::WRL::ComPtr<IPresentationFactory> factory_;
    Microsoft::WRL::ComPtr<IPresentationManager> manager_;
    Microsoft::WRL::ComPtr<IPresentationSurface> surface_;
    Microsoft::WRL::ComPtr<IDCompositionDesktopDevice> dcomp_;
    Microsoft::WRL::ComPtr<IDCompositionTarget> target_;
    Microsoft::WRL::ComPtr<IDCompositionVisual2> visual_;
    Microsoft::WRL::ComPtr<IUnknown> dcomp_surface_;
    HANDLE surface_handle_ = nullptr;
    HANDLE stats_event_ = nullptr;
    HANDLE lost_event_ = nullptr;
    HANDLE fence_event_ = nullptr;
    std::vector<Buffer> buffers_;
    FgD3D12FenceTable fences12_;
    std::vector<Microsoft::WRL::ComPtr<ID3D11Fence>> fences11_;  // index = fence id - 1
    FgClockMap clock_map_;
    FgHighResTimer timer_;

    mutable std::mutex mutex_;
    FgPresentEngine engine_;
    std::vector<FgEnginePresent> deferred_;
    std::vector<Submitted> submitted_;
    std::vector<std::uint32_t> scratch_slots_;
    bool have_vblank_ = false;
    FgTicks last_vblank_ = 0;
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
